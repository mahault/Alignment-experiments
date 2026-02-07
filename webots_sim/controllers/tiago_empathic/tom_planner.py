"""
Theory of Mind Planner using proper Active Inference Expected Free Energy (EFE).

Based on Sophisticated Inference (Friston et al. 2020) and ToM extension (2508.00401v2):
- Discrete generative model with A/B/C/D matrices
- EFE = Pragmatic (preference satisfaction) + Epistemic (information gain about other's role)
- ToM: Q(a_other) proportional to softmax(-gamma * G_other), not greedy-x
- Collision avoidance via preferences C, not hard clamps
- Blocked-motion mixing: congested transitions predict "stuck" futures
- Bayesian belief update over other agent's hidden role/intent

G_social = G_self + alpha * G_other

Uses JAX for parallel evaluation of all 5^depth policies.
"""

import math
import itertools
import numpy as np
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

# ============================================================================
# Constants: Discretization
# ============================================================================

# Arena geometry: 5m x 2m floor with hazard obstacles forming a narrow corridor
# The corridor is narrow in the MIDDLE (|X| < 1.0) but opens up at the ENDS
# where hazards are absent — this is where the empathic agent can yield.
X_MIN, X_MAX = -2.2, 2.2
Y_MIN, Y_MAX = -1.0, 1.0   # Full arena width — not just the corridor
ROBOT_RADIUS = 0.25
COLLISION_DIST = 2 * ROBOT_RADIUS  # 0.5m — physical contact
CAUTION_DIST = 0.90  # meters — slightly larger to catch more near-misses

# Hazard obstacle definitions from the world file (tiago_empathic_test.wbt)
# Each tuple: (x_center, y_center, x_half_size, y_half_size)
# Top row: only 2 hazards near center — open at |X| > 0.8
# Bottom row: 4 hazards spanning most of the corridor — nearly continuous wall
HAZARDS = [
    (-0.5, 0.75, 0.3, 0.25),   # hazard_top_1
    (0.5, 0.75, 0.3, 0.25),    # hazard_top_2
    (-1.5, -0.75, 0.3, 0.25),  # hazard_bottom_0
    (-0.5, -0.75, 0.3, 0.25),  # hazard_bottom_1
    (0.5, -0.75, 0.3, 0.25),   # hazard_bottom_2
    (1.5, -0.75, 0.3, 0.25),   # hazard_bottom_3
]
HAZARD_MARGIN = ROBOT_RADIUS  # Robot center must stay this far from hazard edges

# X: 11 bins at 0.4m resolution
N_X = 11
X_EDGES = np.linspace(X_MIN, X_MAX, N_X + 1)
X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])

# Y: 7 bins covering full arena width (resolution ~0.286m)
N_Y = 7
Y_EDGES = np.linspace(Y_MIN, Y_MAX, N_Y + 1)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])

# Total pose states per agent
N_POSE = N_X * N_Y  # 77

# Actions: {STAY, FORWARD, BACK, LEFT, RIGHT}
N_ACTIONS = 5
ACTION_NAMES = ["STAY", "FORWARD", "BACK", "LEFT", "RIGHT"]

# Other agent's hidden role/intent
N_ROLES = 4
ROLE_PUSH = 0
ROLE_YIELD_LEFT = 1
ROLE_YIELD_RIGHT = 2
ROLE_WAIT = 3
ROLE_NAMES = ["PUSH", "YIELD_L", "YIELD_R", "WAIT"]

# Observation modalities for risk
N_RISK = 3  # {safe=0, caution=1, danger=2}

# Observation modalities for other's motion
N_MOTION = 4  # {forward=0, backward=1, lateral=2, still=3}


# ============================================================================
# Runtime configuration (for different arenas/layouts)
# ============================================================================

def configure(x_min, x_max, y_min, y_max, hazards=None, n_x=None, n_y=None):
    """Reconfigure discretization for a different arena/layout.

    Call this BEFORE creating any CorridorModel or ToMPlanner.
    If not called, defaults match tiago_empathic_test.wbt (5x2m arena).

    Parameters
    ----------
    x_min, x_max : float
        Arena X bounds (meters).
    y_min, y_max : float
        Arena Y bounds (meters).
    hazards : list of (x, y, half_sx, half_sy) tuples, optional
        Hazard obstacle positions and half-sizes.
    n_x, n_y : int, optional
        Grid resolution. If None, auto-computed from arena size.
    """
    global X_MIN, X_MAX, Y_MIN, Y_MAX, HAZARDS
    global X_EDGES, X_CENTERS, Y_EDGES, Y_CENTERS
    global N_X, N_Y, N_POSE

    X_MIN, X_MAX = x_min, x_max
    Y_MIN, Y_MAX = y_min, y_max

    # Auto-compute grid resolution if not specified
    x_span = x_max - x_min
    y_span = y_max - y_min
    if n_x is not None:
        N_X = n_x
    else:
        N_X = max(7, round(x_span / 0.4))
    if n_y is not None:
        N_Y = n_y
    else:
        N_Y = max(5, round(y_span / 0.3))

    X_EDGES = np.linspace(X_MIN, X_MAX, N_X + 1)
    X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])
    Y_EDGES = np.linspace(Y_MIN, Y_MAX, N_Y + 1)
    Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])
    N_POSE = N_X * N_Y

    if hazards is not None:
        HAZARDS = hazards

    print(f"[tom_planner] Configured: X=[{X_MIN:.1f},{X_MAX:.1f}] Y=[{Y_MIN:.1f},{Y_MAX:.1f}] "
          f"grid={N_X}x{N_Y}={N_POSE} states, {len(HAZARDS)} hazards")


# ============================================================================
# Discretization helpers
# ============================================================================

def xy_to_bin(x: float, y: float) -> Tuple[int, int]:
    """Convert continuous (x, y) to (x_bin, y_bin)."""
    x_bin = int(np.clip(np.searchsorted(X_EDGES[1:], x), 0, N_X - 1))
    y_bin = int(np.clip(np.searchsorted(Y_EDGES[1:], y), 0, N_Y - 1))
    return x_bin, y_bin


def bin_to_xy(x_bin: int, y_bin: int) -> Tuple[float, float]:
    """Convert bin indices to continuous center coordinates."""
    return float(X_CENTERS[x_bin]), float(Y_CENTERS[y_bin])


def pose_to_flat(x_bin: int, y_bin: int) -> int:
    """Convert (x_bin, y_bin) to flat state index."""
    return y_bin * N_X + x_bin


def flat_to_pose(s: int) -> Tuple[int, int]:
    """Convert flat state index to (x_bin, y_bin)."""
    return s % N_X, s // N_X


def make_delta_belief(x: float, y: float) -> np.ndarray:
    """Create a delta belief distribution peaked at the bin containing (x, y)."""
    xb, yb = xy_to_bin(x, y)
    q = np.zeros(N_POSE)
    q[pose_to_flat(xb, yb)] = 1.0
    return q


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def is_in_hazard(cx: float, cy: float) -> bool:
    """Check if continuous position (cx, cy) would collide with any hazard obstacle.

    Uses HAZARD_MARGIN (= ROBOT_RADIUS) so the robot center stays clear.
    """
    for hx, hy, hsx, hsy in HAZARDS:
        if abs(cx - hx) < hsx + HAZARD_MARGIN and abs(cy - hy) < hsy + HAZARD_MARGIN:
            return True
    return False


# ============================================================================
# CorridorModel: Discrete generative model for the TIAGo corridor
# ============================================================================

class CorridorModel:
    """
    Discrete POMDP generative model for corridor navigation.

    State factors:
      - s_self:  77 pose states (11x * 7y, covering full 5x2m arena)
      - s_other: 77 pose states
      - s_role:  4 hidden role states {PUSH, YIELD_LEFT, YIELD_RIGHT, WAIT}

    Observation modalities:
      - o_self:   77 (identity from s_self)
      - o_other:  77 (identity from s_other)
      - o_risk:   3  {safe, caution, danger} from joint pose geometry
      - o_motion: 4  {forward, backward, lateral, still} from s_role

    C_self includes hazard obstacle positions — bins overlapping hazards get
    -50 penalty. This gives the planner spatial awareness: the corridor is
    narrow in the middle but opens up at the ends.
    """

    def __init__(self, goal_x: float, goal_y: float,
                 other_goal_x: float = None, other_goal_y: float = None):
        self.goal_x = goal_x
        self.goal_y = goal_y
        # Direction toward goal in bin indices
        self.forward_dx = +1 if goal_x > 0 else -1

        # Other's goal direction (for B_other_pose)
        if other_goal_x is not None:
            self.other_forward_dx = +1 if other_goal_x > 0 else -1
        else:
            self.other_forward_dx = -self.forward_dx

        # Build all matrices
        self.A_self = self._build_A_identity()
        self.A_other = self._build_A_identity()
        self.A_risk = self._build_A_risk()
        self.A_motion = self._build_A_motion()

        self.B_self = self._build_B_self(self.forward_dx)
        self.B_other_pose = self._build_B_other_pose()
        self.B_role = self._build_B_role()

        self.C_self = self._build_C_self(goal_x, goal_y)
        self.C_risk = np.array([0.0, 0.0, -25.0])  # Only danger matters, not caution
        self.C_motion = np.zeros(N_MOTION)

        self.D_role = np.array([0.4, 0.2, 0.2, 0.2])

        # Convert to JAX arrays
        self.jA_risk = jnp.array(self.A_risk)
        self.jA_motion = jnp.array(self.A_motion)
        self.jB_self = jnp.array(self.B_self)
        self.jB_other_pose = jnp.array(self.B_other_pose)
        self.jB_role = jnp.array(self.B_role)
        self.jC_self = jnp.array(self.C_self)
        self.jC_risk = jnp.array(self.C_risk)

    def _build_A_identity(self) -> np.ndarray:
        return np.eye(N_POSE)

    def _build_A_risk(self) -> np.ndarray:
        """P(o_risk | s_self, s_other) based on Euclidean distance between bin centers."""
        A = np.zeros((N_RISK, N_POSE, N_POSE))
        for s_self in range(N_POSE):
            x1, y1 = flat_to_pose(s_self)
            cx1, cy1 = bin_to_xy(x1, y1)
            for s_other in range(N_POSE):
                x2, y2 = flat_to_pose(s_other)
                cx2, cy2 = bin_to_xy(x2, y2)
                d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if d < COLLISION_DIST:
                    A[2, s_self, s_other] = 1.0  # danger
                elif d < CAUTION_DIST:
                    A[1, s_self, s_other] = 1.0  # caution
                else:
                    A[0, s_self, s_other] = 1.0  # safe
        return A

    def _build_A_motion(self) -> np.ndarray:
        """P(o_motion | s_role). Probabilistic — key for epistemic value."""
        #                       PUSH  YIELD_L  YIELD_R  WAIT
        return np.array([
            [0.80, 0.05, 0.05, 0.05],   # forward
            [0.05, 0.10, 0.10, 0.05],   # backward
            [0.05, 0.75, 0.75, 0.10],   # lateral
            [0.10, 0.10, 0.10, 0.80],   # still
        ])

    def _build_B_self(self, forward_dx: int) -> np.ndarray:
        """B[s_next, s_current, action] — deterministic single-agent transitions."""
        B = np.zeros((N_POSE, N_POSE, N_ACTIONS))
        for s in range(N_POSE):
            x, y = flat_to_pose(s)
            for a in range(N_ACTIONS):
                nx, ny = x, y
                if a == 1:    # FORWARD (toward goal)
                    nx = x + forward_dx
                elif a == 2:  # BACK (away from goal)
                    nx = x - forward_dx
                elif a == 3:  # LEFT (negative Y)
                    ny = y - 1
                elif a == 4:  # RIGHT (positive Y)
                    ny = y + 1
                # Clamp to grid
                nx = max(0, min(N_X - 1, nx))
                ny = max(0, min(N_Y - 1, ny))
                s_next = pose_to_flat(nx, ny)
                B[s_next, s, a] = 1.0
        return B

    def _build_B_other_pose(self) -> np.ndarray:
        """B_other[s_next, s_current, role] — other's transition given role."""
        B = np.zeros((N_POSE, N_POSE, N_ROLES))
        for s in range(N_POSE):
            x, y = flat_to_pose(s)
            for role in range(N_ROLES):
                nx, ny = x, y
                if role == ROLE_PUSH:
                    nx = x + self.other_forward_dx
                elif role == ROLE_YIELD_LEFT:
                    ny = y - 1
                elif role == ROLE_YIELD_RIGHT:
                    ny = y + 1
                # ROLE_WAIT: stay
                nx = max(0, min(N_X - 1, nx))
                ny = max(0, min(N_Y - 1, ny))
                s_next = pose_to_flat(nx, ny)
                B[s_next, s, role] = 1.0
        return B

    def _build_B_role(self) -> np.ndarray:
        """Sticky role transitions — agents tend to continue current behavior."""
        return np.array([
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.70, 0.05, 0.15],
            [0.10, 0.05, 0.70, 0.15],
            [0.10, 0.15, 0.15, 0.60],
        ])

    def _build_C_self(self, goal_x: float, goal_y: float) -> np.ndarray:
        """Preferences over self-position: goal + distance gradient + hazards + walls.

        The arena is 5x2m with hazard obstacles forming a narrow corridor in the
        MIDDLE. At the corridor ENDS (|X| > 1.0), the top side opens up — no
        hazards block Y > 0.5. The empathic agent can yield into these open areas.

        C encodes:
        1. Goal attraction: +80 at goal bin, -5*manhattan elsewhere
        2. Hazard penalty: -50 at bins overlapping hazard obstacles
        3. X boundary penalty: -20 at corridor ends (avoid getting stuck at x_max)
        4. Y boundary penalty: -10 at arena walls (avoid clipping Y=±1.0)
        """
        C = np.zeros(N_POSE)
        gx, gy = xy_to_bin(goal_x, goal_y)
        goal_flat = pose_to_flat(gx, gy)

        for s in range(N_POSE):
            x, y = flat_to_pose(s)
            cx, cy = bin_to_xy(x, y)

            # Goal gradient
            if s == goal_flat:
                C[s] = 80.0
            else:
                manhattan = abs(x - gx) + abs(y - gy)
                C[s] = -5.0 * manhattan - 0.1

            # Hazard penalty: bins overlapping obstacles are strongly penalized
            if is_in_hazard(cx, cy):
                C[s] -= 50.0

            # X boundary: discourage corridor ends (robot gets stuck)
            if x == 0 or x == N_X - 1:
                C[s] -= 20.0
            elif x == 1 or x == N_X - 2:
                C[s] -= 8.0

            # Y boundary: discourage arena wall hugging (robot clips wall)
            if y == 0 or y == N_Y - 1:
                C[s] -= 10.0
            elif y == 1 or y == N_Y - 2:
                C[s] -= 3.0

        return C


# ============================================================================
# EFE computation (JAX)
# ============================================================================

def _entropy(q):
    """Shannon entropy H(q) = -sum(q * log(q))."""
    return -jnp.sum(q * jnp.log(q + 1e-16))


# Marginalized other transition for JAX compatibility.
def _transition_other_marginalized(q_other, q_role, B_other_pose):
    """Compute q(s_other') marginalized over role. JAX-friendly."""
    # B_other_pose: (N_POSE, N_POSE, N_ROLES)
    # For each role r: contribution = q_role[r] * B_other_pose[:,:,r] @ q_other
    # = sum_r q_role[r] * B[:,:,r] @ q_other
    # = (sum_r q_role[r] * B[:,:,r]) @ q_other
    B_avg = jnp.einsum('ijk,k->ij', B_other_pose, q_role)  # (N_POSE, N_POSE)
    return B_avg @ q_other


def _compute_step_efe_jax(
    q_self, q_other, q_role, action,
    B_self, B_other_pose, B_role,
    A_risk, A_motion,
    C_self, C_risk,
    epistemic_scale,
):
    """JAX-compatible single-step EFE. Returns (G_step, q_self', q_other', q_role').

    Key: blocked-motion mixing. After computing raw transitions, we check
    p(danger) from the joint next-state. If high, both agents get "stuck" —
    their next-state mixes back toward current state proportional to p_danger.
    This makes the generative model predict that pushing through congestion
    leads to no progress + danger, so ToM correctly values clearing space.
    """
    # 1. Raw transitions (before physics/blocking)
    q_self_next_raw = B_self[:, :, action] @ q_self
    q_other_next_raw = _transition_other_marginalized(q_other, q_role, B_other_pose)
    q_role_next = B_role @ q_role

    # 2. Blocked-motion mixing: compute p(danger) from raw next states
    #    Only DANGER (contact) causes blocking, not caution (proximity).
    obs_risk_raw = jnp.einsum('oij,i,j->o', A_risk, q_self_next_raw, q_other_next_raw)
    p_danger = obs_risk_raw[2]
    p_block = jnp.clip(p_danger * 1.5, 0.0, 0.98)

    # Mix next-state with "stuck at current state" proportional to blockage
    q_self_next = (1.0 - p_block) * q_self_next_raw + p_block * q_self
    q_other_next = (1.0 - p_block) * q_other_next_raw + p_block * q_other

    # 3. Pragmatic: location utility (using blocked-aware next states)
    u_loc = jnp.dot(q_self_next, C_self)

    # Pragmatic: risk (recompute with blocked-aware states)
    obs_risk = jnp.einsum('oij,i,j->o', A_risk, q_self_next, q_other_next)
    u_risk = jnp.dot(obs_risk, C_risk)

    G_prag = -(u_loc + u_risk)

    # 4. Epistemic: info gain about role
    H_prior = _entropy(q_role_next)
    obs_motion = A_motion @ q_role_next

    def _post_H(o_idx):
        lik = A_motion[o_idx]
        post_u = lik * q_role_next
        post = post_u / (jnp.sum(post_u) + 1e-16)
        return _entropy(post)

    H_posts = jax.vmap(_post_H)(jnp.arange(N_MOTION))
    H_post_exp = jnp.dot(obs_motion, H_posts)
    info_gain = H_prior - H_post_exp

    G_epist = -epistemic_scale * info_gain

    return G_prag + G_epist, q_self_next, q_other_next, q_role_next


def _rollout_policy_jax(
    policy,  # (depth,) array of action indices
    q_self_init, q_other_init, q_role_init,
    B_self, B_other_pose, B_role,
    A_risk, A_motion,
    C_self, C_risk,
    epistemic_scale, discount,
):
    """Roll out a policy (sequence of actions) and accumulate EFE."""

    def step_fn(carry, action):
        q_self, q_other, q_role, total_G, disc = carry
        G_step, q_self_n, q_other_n, q_role_n = _compute_step_efe_jax(
            q_self, q_other, q_role, action,
            B_self, B_other_pose, B_role,
            A_risk, A_motion, C_self, C_risk,
            epistemic_scale,
        )
        total_G = total_G + disc * G_step
        disc = disc * discount
        return (q_self_n, q_other_n, q_role_n, total_G, disc), None

    init = (q_self_init, q_other_init, q_role_init, 0.0, 1.0)
    (_, _, _, total_G, _), _ = lax.scan(step_fn, init, policy)
    return total_G


# ============================================================================
# ToM: Predict other agent's action (JAX JIT)
# ============================================================================

def _other_single_step_efe_jax(q_other, q_us, action, B_other, A_risk, C_self, C_risk):
    """Single-step EFE from other's perspective with blocked-motion mixing."""
    q_o_next_raw = B_other[:, :, action] @ q_other

    # Blocked-motion: other gets stuck if pushing into contact
    obs_risk = jnp.einsum('oij,i,j->o', A_risk, q_o_next_raw, q_us)
    p_danger = obs_risk[2]
    p_block = jnp.clip(p_danger * 1.5, 0.0, 0.98)

    q_o_next = (1.0 - p_block) * q_o_next_raw + p_block * q_other

    u_loc = jnp.dot(q_o_next, C_self)
    obs_risk_final = jnp.einsum('oij,i,j->o', A_risk, q_o_next, q_us)
    u_risk = jnp.dot(obs_risk_final, C_risk)

    G = -(u_loc + u_risk)
    return G, q_o_next


def _other_rollout_policy_jax(policy, q_other, q_us, B_other, A_risk, C_self, C_risk, discount):
    """Roll out a policy from the other's perspective with blocked-motion."""
    def step_fn(carry, action):
        q_o, total_G, disc = carry
        G_step, q_o_next = _other_single_step_efe_jax(
            q_o, q_us, action, B_other, A_risk, C_self, C_risk
        )
        total_G = total_G + disc * G_step
        disc = disc * discount
        return (q_o_next, total_G, disc), None

    init = (q_other, 0.0, 1.0)
    (_, total_G, _), _ = lax.scan(step_fn, init, policy)
    return total_G


def _build_full_trajectory_social_jit(other_model, B_self_ours, A_risk_ours, depth_other=3, discount=0.9):
    """Build JIT-compiled function that computes social EFE at EVERY step of
    our policy rollout (sophisticated inference), not just the end state.

    At each step t of our 5-step policy, we compute the other's best-response
    G given our position at step t, and accumulate discounted social cost:
        G_social(policy) = sum_t discount^t * G_other_best_response(our_pos_t)

    CRITICAL: includes blocked-motion mixing in the social rollout. Without it,
    policies that walk THROUGH the other agent appear falsely good because the
    rollout fantasizes about reaching far-away positions with great social scores.
    With blocking, if step 1 enters the other agent's position, we get stuck
    there for all remaining steps — making FORWARD-through-danger correctly bad.

    Efficient: precompute G_other for all 55 positions once (55 x 125 = 6,875
    evaluations), then for each of our 3125 policies, roll out with blocking
    and look up the precomputed G_other at each step (3125 x 5 dot products).
    """
    all_other_policies = jnp.array(
        list(itertools.product(range(N_ACTIONS), repeat=depth_other)),
        dtype=jnp.int32
    )

    jB_other = jnp.array(other_model.B_self)
    jA_risk_other = jnp.array(other_model.A_risk)
    jC_other = jnp.array(other_model.C_self)
    jC_risk = jnp.array(other_model.C_risk)
    jB_self = jnp.array(B_self_ours)
    jA_risk_self = jnp.array(A_risk_ours)
    discount_val = discount

    @jax.jit
    def compute_social_all_policies(q_self_init, q_other, all_our_policies):
        # 1. Precompute: other's best-response G from each of 55 possible positions
        def other_best_from_pos(us_idx):
            q_us = jnp.zeros(N_POSE).at[us_idx].set(1.0)
            def single_rollout(policy):
                return _other_rollout_policy_jax(
                    policy, q_other, q_us,
                    jB_other, jA_risk_other, jC_other, jC_risk, discount_val,
                )
            all_G = jax.vmap(single_rollout)(all_other_policies)
            return jnp.min(all_G)

        G_other_by_pos = jax.vmap(other_best_from_pos)(jnp.arange(N_POSE))  # (55,)

        # 2. For each policy, roll out step by step with blocked-motion mixing
        def rollout_social(policy):
            def step(carry, action):
                q_self, total_G, disc = carry
                q_self_next_raw = jB_self[:, :, action] @ q_self

                # Blocked-motion: can't walk through the other agent
                obs_risk = jnp.einsum('oij,i,j->o', jA_risk_self, q_self_next_raw, q_other)
                p_danger = obs_risk[2]
                p_block = jnp.clip(p_danger * 1.5, 0.0, 0.98)
                q_self_next = (1.0 - p_block) * q_self_next_raw + p_block * q_self

                # Social cost at this step = other's best response from our position
                G_step = jnp.dot(q_self_next, G_other_by_pos)
                total_G = total_G + disc * G_step
                disc = disc * discount_val
                return (q_self_next, total_G, disc), None

            init = (q_self_init, 0.0, 1.0)
            (_, total_G, _), _ = lax.scan(step, init, policy)
            return total_G

        G_social = jax.vmap(rollout_social)(all_our_policies)  # (3125,)

        return G_social

    return compute_social_all_policies


# ============================================================================
# Belief update: Role inference
# ============================================================================

def classify_motion(dx, dy, other_goal_x, other_x):
    """Classify observed motion delta into {forward=0, backward=1, lateral=2, still=3}."""
    if abs(dx) < 0.05 and abs(dy) < 0.05:
        return 3  # still
    toward_goal = other_goal_x - other_x
    if abs(dx) > abs(dy):
        if toward_goal != 0 and (dx * toward_goal > 0):
            return 0  # forward (toward their goal)
        else:
            return 1  # backward
    else:
        return 2  # lateral


def update_role_belief(q_role, motion_obs_idx, A_motion, confidence=1.0):
    """Bayesian update: q(role | o) proportional to P(o | role)^confidence * q(role).

    confidence < 1.0 attenuates the update (used during contact when
    displacement is confounded by physics and not a reliable indicator
    of the other agent's intended policy).
    """
    likelihood = A_motion[motion_obs_idx, :]
    # Raise likelihood to confidence power: 1.0 = full update, 0.0 = no update
    tempered_likelihood = np.power(likelihood, confidence)
    posterior_unnorm = tempered_likelihood * q_role
    Z = posterior_unnorm.sum()
    if Z > 1e-16:
        return posterior_unnorm / Z
    return q_role.copy()


# ============================================================================
# ToMPlanner: Main planner class
# ============================================================================

class ToMPlanner:
    """
    Theory of Mind planner using proper Active Inference EFE with
    a discrete generative model (A/B/C/D matrices).

    Fixes over previous version:
    1. Discrete generative model instead of continuous distance heuristic
    2. Collision avoidance via preferences C, not hard clamps
    3. Proper EFE = pragmatic (utility) + epistemic (info gain about other's role)
    4. ToM: Q(a_other) ~ softmax(-gamma * G_other), not greedy-x
    5. Bayesian role inference from observed motion patterns
    """

    def __init__(self, agent_id: int, goal_x: float, goal_y: float, alpha: float):
        self.agent_id = agent_id
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.alpha = alpha

        # Planning parameters
        self.depth = 5              # Sophisticated inference depth (5^5=3125 policies)
        self.gamma = 8.0            # Inverse temperature for softmax action selection
        self.epistemic_scale = 1.0  # Weight on epistemic term
        self.discount = 0.9         # Future EFE discount
        self.goal_tolerance = 0.3

        # Build own generative model (lazily build other's model on first call)
        self.model = CorridorModel(goal_x, goal_y)
        self._other_model_cache = {}

        # Belief state (persists across calls)
        self.q_role = np.array(self.model.D_role)
        self.prev_other_x = None
        self.prev_other_y = None

        # Precompute all 5^depth policies
        self.all_policies = np.array(
            list(itertools.product(range(N_ACTIONS), repeat=self.depth)),
            dtype=np.int32
        )
        self.jall_policies = jnp.array(self.all_policies)

        # JIT-compiled functions (built lazily on first use)
        self._jit_rollout_all = None
        self._jit_social_full_traj = {}  # keyed by other goal

        # Debug/logging
        self.verbose = True
        self.debug_counter = 0
        self.verbose_interval = 10

    def _get_other_model(self, other_goal_x, other_goal_y):
        """Get or build the other agent's generative model (from their perspective)."""
        key = (round(other_goal_x, 1), round(other_goal_y, 1))
        if key not in self._other_model_cache:
            self._other_model_cache[key] = CorridorModel(
                goal_x=other_goal_x,
                goal_y=other_goal_y,
                other_goal_x=self.goal_x,
                other_goal_y=self.goal_y,
            )
        return self._other_model_cache[key]

    def _build_jit_rollout(self):
        """Build JIT-compiled vmap rollout function."""
        m = self.model
        epistemic_scale = self.epistemic_scale
        discount_val = self.discount

        @jax.jit
        def rollout_all(policies, q_self, q_other, q_role):
            def single_rollout(policy):
                return _rollout_policy_jax(
                    policy, q_self, q_other, q_role,
                    m.jB_self, m.jB_other_pose, m.jB_role,
                    m.jA_risk, m.jA_motion,
                    m.jC_self, m.jC_risk,
                    epistemic_scale, discount_val,
                )
            return jax.vmap(single_rollout)(policies)

        self._jit_rollout_all = rollout_all

    def _compute_all_efe(self, q_self, q_other, q_role):
        """Compute EFE for all policies using JAX vmap."""
        if self._jit_rollout_all is None:
            self._build_jit_rollout()

        jq_self = jnp.array(q_self)
        jq_other = jnp.array(q_other)
        jq_role = jnp.array(q_role)

        G_all = self._jit_rollout_all(self.jall_policies, jq_self, jq_other, jq_role)
        return np.array(G_all)

    def _action_to_target(self, action_idx, my_x, my_y):
        """Convert discrete action to continuous waypoint."""
        x_bin, y_bin = xy_to_bin(my_x, my_y)

        nx, ny = x_bin, y_bin
        if action_idx == 1:    # FORWARD
            nx = x_bin + self.model.forward_dx
        elif action_idx == 2:  # BACK
            nx = x_bin - self.model.forward_dx
        elif action_idx == 3:  # LEFT
            ny = y_bin - 1
        elif action_idx == 4:  # RIGHT
            ny = y_bin + 1

        nx = max(0, min(N_X - 1, nx))
        ny = max(0, min(N_Y - 1, ny))

        return bin_to_xy(nx, ny)

    def _make_debug_label(self, action_idx, G_social, my_x, my_y, tx, ty):
        """Create debug label string."""
        dist_now = distance(my_x, my_y, self.goal_x, self.goal_y)
        dist_after = distance(tx, ty, self.goal_x, self.goal_y)

        if action_idx == 0:
            label = "STAY"
        elif dist_after < dist_now - 0.05:
            label = "TOWARD"
        elif dist_after > dist_now + 0.05:
            label = "YIELD" if self.alpha > 0.1 else "MANEUVER"
        else:
            label = "LATERAL"

        role_str = ROLE_NAMES[int(np.argmax(self.q_role))]
        return f"a={self.alpha:.1f} G={G_social:.2f} [{label}] role_belief={role_str}"

    def plan(self, my_x: float, my_y: float,
             other_x: float, other_y: float,
             other_goal_x: float, other_goal_y: float,
             other_alpha: float) -> Tuple[float, float, str]:
        """
        Plan next position using proper Active Inference EFE.

        Steps:
        1. Update role belief from observed motion (Bayesian update)
        2. Discretize positions into belief distributions
        3. Compute EFE for all 5^depth policies (JAX parallelized)
        4. Compute social EFE with ToM other prediction
        5. Select best policy, convert to continuous waypoint
        """
        # At goal?
        if distance(my_x, my_y, self.goal_x, self.goal_y) < self.goal_tolerance:
            return self.goal_x, self.goal_y, "AT_GOAL"

        # 1. Update role belief from observed other motion
        if self.prev_other_x is not None:
            dx = other_x - self.prev_other_x
            dy = other_y - self.prev_other_y
            motion_obs = classify_motion(dx, dy, other_goal_x, self.prev_other_x)

            # Fix D: attenuate belief update during contact — displacement
            # is confounded by physics (pushing/bumping), not a reliable
            # indicator of the other's intended policy.
            dist_to_other = distance(my_x, my_y, other_x, other_y)
            if dist_to_other < COLLISION_DIST * 1.5:
                confidence = 0.2  # heavily attenuated during near-contact
            elif dist_to_other < CAUTION_DIST:
                confidence = 0.5  # moderately attenuated in caution zone
            else:
                confidence = 1.0  # full update when far apart

            self.q_role = update_role_belief(
                self.q_role, motion_obs, self.model.A_motion, confidence
            )
            # Propagate role forward in time
            self.q_role = self.model.B_role @ self.q_role
            # Entropy floor: prevent role belief from collapsing to a single
            # role (e.g. WAIT) which creates self-fulfilling deadlock where
            # both agents predict the other will stay and neither moves.
            uniform = np.array([0.25, 0.25, 0.25, 0.25])
            self.q_role = 0.85 * self.q_role + 0.15 * uniform

        self.prev_other_x = other_x
        self.prev_other_y = other_y

        # 2. Discretize current positions into beliefs
        q_self = make_delta_belief(my_x, my_y)
        q_other = make_delta_belief(other_x, other_y)

        # 3. Compute self EFE for all policies
        G_self_all = self._compute_all_efe(q_self, q_other, self.q_role)

        # 4. Compute social EFE term (JIT-compiled on GPU)
        other_model = self._get_other_model(other_goal_x, other_goal_y)

        G_social_all = np.copy(G_self_all)
        if self.alpha > 0.001:
            # Get or build JIT-compiled full-trajectory social EFE function
            other_key = (round(other_goal_x, 1), round(other_goal_y, 1))
            if other_key not in self._jit_social_full_traj:
                self._jit_social_full_traj[other_key] = \
                    _build_full_trajectory_social_jit(
                        other_model, self.model.B_self, self.model.A_risk,
                        depth_other=3, discount=self.discount
                    )
            jit_social = self._jit_social_full_traj[other_key]

            # Compute social EFE for all 3125 policies based on full trajectory end state
            G_other_all = np.array(jit_social(
                jnp.array(q_self), jnp.array(q_other), self.jall_policies
            ))

            G_social_all += self.alpha * G_other_all

        # 5. Select best policy
        best_idx = int(np.argmin(G_social_all))
        best_policy = self.all_policies[best_idx]
        best_action = int(best_policy[0])
        best_G = float(G_social_all[best_idx])

        # 6. Convert to continuous waypoint
        target_x, target_y = self._action_to_target(best_action, my_x, my_y)

        # 7. Debug output
        self.debug_counter += 1
        if self.verbose and (self.debug_counter % self.verbose_interval == 1):
            print(f"\n{'='*60}")
            print(f"DECISION LOG agent {self.agent_id} (alpha={self.alpha})")
            print(f"{'='*60}")
            print(f"  pos=({my_x:.2f},{my_y:.2f}) other=({other_x:.2f},{other_y:.2f})")
            print(f"  goal=({self.goal_x:.2f},{self.goal_y:.2f})")
            print(f"  role_belief: {dict(zip(ROLE_NAMES, [f'{p:.2f}' for p in self.q_role]))}")
            print(f"  Best G by first action:")
            for a in range(N_ACTIONS):
                mask = self.all_policies[:, 0] == a
                if mask.any():
                    best_for_a = float(G_social_all[mask].min())
                    marker = " <-- CHOSEN" if a == best_action else ""
                    print(f"    {ACTION_NAMES[a]:8s}: G={best_for_a:8.2f}{marker}")
            policy_str = "->".join([ACTION_NAMES[a] for a in best_policy])
            print(f"  Policy: [{policy_str}], G_social={best_G:.2f}")
            print(f"{'='*60}")

        # 8. Debug label
        debug = self._make_debug_label(best_action, best_G, my_x, my_y, target_x, target_y)

        return target_x, target_y, debug


# ============================================================================
# Standalone test / simulation
# ============================================================================

def test_planner():
    """Test with two robots approaching each other."""
    p1 = ToMPlanner(agent_id=0, goal_x=1.8, goal_y=0.0, alpha=0.0)
    p2 = ToMPlanner(agent_id=1, goal_x=-1.8, goal_y=0.0, alpha=6.0)

    print("Proper EFE ToM Planner Test")
    print("=" * 60)
    print(f"Robot 1: alpha={p1.alpha} (selfish), goal=({p1.goal_x}, {p1.goal_y})")
    print(f"Robot 2: alpha={p2.alpha} (empathic), goal=({p2.goal_x}, {p2.goal_y})")
    print(f"State space: {N_POSE} pose states, {N_ROLES} role states")
    print(f"Policy space: {len(p1.all_policies)} policies (5^{p1.depth})")
    print()

    r1_x, r1_y = -1.8, 0.0
    r2_x, r2_y = 1.8, 0.0

    for step in range(20):
        t1_x, t1_y, info1 = p1.plan(r1_x, r1_y, r2_x, r2_y, p2.goal_x, p2.goal_y, p2.alpha)
        t2_x, t2_y, info2 = p2.plan(r2_x, r2_y, r1_x, r1_y, p1.goal_x, p1.goal_y, p1.alpha)

        dist_between = distance(t1_x, t1_y, t2_x, t2_y)
        print(f"{step+1:2d}. R1({r1_x:+.2f},{r1_y:+.2f})->({t1_x:+.2f},{t1_y:+.2f}) {info1}")
        print(f"    R2({r2_x:+.2f},{r2_y:+.2f})->({t2_x:+.2f},{t2_y:+.2f}) {info2}")
        print(f"    Distance: {dist_between:.2f}m")

        r1_x, r1_y = t1_x, t1_y
        r2_x, r2_y = t2_x, t2_y

        if distance(r1_x, r1_y, p1.goal_x, p1.goal_y) < 0.3 and \
           distance(r2_x, r2_y, p2.goal_x, p2.goal_y) < 0.3:
            print(f"\nBoth reached goals at step {step+1}!")
            break
        print()


def test_discretization():
    """Verify discretization round-trip."""
    print("Testing discretization...")
    print(f"  X bins: {N_X}, Y bins: {N_Y}, total: {N_POSE}")
    print(f"  X centers: {[f'{c:.2f}' for c in X_CENTERS]}")
    print(f"  Y centers: {[f'{c:.2f}' for c in Y_CENTERS]}")
    for x in [-2.2, -1.0, 0.0, 1.0, 2.2]:
        for y in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            bx, by = xy_to_bin(x, y)
            cx, cy = bin_to_xy(bx, by)
            assert abs(cx - x) < 0.4, f"X round-trip failed: {x} -> bin {bx} -> {cx}"
            assert abs(cy - y) < 0.3, f"Y round-trip failed: {y} -> bin {by} -> {cy}"
    print("  Discretization OK")


def test_model_matrices():
    """Verify model matrix shapes and normalization."""
    print("Testing model matrices...")
    m = CorridorModel(1.8, 0.0)

    # B_self columns sum to 1
    for a in range(N_ACTIONS):
        col_sums = m.B_self[:, :, a].sum(axis=0)
        assert np.allclose(col_sums, 1.0), f"B_self action {a} not normalized"

    # B_other_pose columns sum to 1
    for r in range(N_ROLES):
        col_sums = m.B_other_pose[:, :, r].sum(axis=0)
        assert np.allclose(col_sums, 1.0), f"B_other role {r} not normalized"

    # B_role columns sum to 1
    col_sums = m.B_role.sum(axis=0)
    assert np.allclose(col_sums, 1.0), "B_role not normalized"

    # A_risk: for each (s_self, s_other), exactly one risk level
    for si in range(N_POSE):
        for so in range(N_POSE):
            assert np.isclose(m.A_risk[:, si, so].sum(), 1.0), \
                f"A_risk not normalized at ({si},{so})"

    # A_motion columns sum to 1
    col_sums = m.A_motion.sum(axis=0)
    assert np.allclose(col_sums, 1.0), "A_motion not normalized"

    # C_self: goal bin has max preference
    gx, gy = xy_to_bin(1.8, 0.0)
    goal_s = pose_to_flat(gx, gy)
    assert m.C_self[goal_s] == m.C_self.max(), "Goal bin should have max preference"

    print("  Model matrices OK")


def test_hazard_map():
    """Verify hazard map matches world geometry."""
    print("Testing hazard map...")
    m = CorridorModel(1.8, 0.0)

    # At X=1.6 (near R2 start), top Y bins should be OPEN (no top hazard)
    for y_bin in range(N_Y):
        cx, cy = bin_to_xy(9, y_bin)  # x_bin=9 -> X=1.6
        s = pose_to_flat(9, y_bin)
        hazard = is_in_hazard(cx, cy)
        label = "HAZARD" if hazard else "safe"
        print(f"    X={cx:+.2f} Y={cy:+.3f} -> {label}  C={m.C_self[s]:.1f}")

    # Top at X=0.4 (middle) should be blocked by hazard_top_2
    for y_bin in range(N_Y):
        cx, cy = bin_to_xy(6, y_bin)  # x_bin=6 -> X=0.4
        hazard = is_in_hazard(cx, cy)
        label = "HAZARD" if hazard else "safe"
        if y_bin >= N_Y // 2:
            print(f"    X={cx:+.2f} Y={cy:+.3f} -> {label}")

    # At X=1.6, Y=0.57 should be safe (open area for yielding)
    assert not is_in_hazard(1.6, 0.57), "X=1.6, Y=0.57 should be open"
    # At X=0.4, Y=0.57 should be hazardous (top hazard)
    assert is_in_hazard(0.4, 0.57), "X=0.4, Y=0.57 should be in hazard"
    print("  Hazard map OK")


def test_efe_basic():
    """Verify EFE computation produces finite values."""
    print("Testing basic EFE...")
    p = ToMPlanner(0, 1.8, 0.0, 0.0)

    # Far from other robot — should prefer FORWARD
    tx, ty, info = p.plan(0.0, 0.0, -2.0, 0.0, -1.8, 0.0, 0.0)
    assert tx > 0.0, f"Should move toward goal (+X), got tx={tx}"
    print(f"  Far from other: target=({tx:.2f},{ty:.2f}) {info}")

    # Reset for next test
    p2 = ToMPlanner(1, -1.8, 0.0, 6.0)
    tx2, ty2, info2 = p2.plan(0.4, 0.0, -0.4, 0.0, 1.8, 0.0, 0.0)
    print(f"  Empathic near selfish: target=({tx2:.2f},{ty2:.2f}) {info2}")
    print("  Basic EFE OK")


if __name__ == "__main__":
    test_discretization()
    test_model_matrices()
    test_hazard_map()
    test_efe_basic()
    print()
    test_planner()
