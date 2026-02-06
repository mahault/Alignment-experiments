"""
Theory of Mind Planner using proper Active Inference Expected Free Energy (EFE).

Based on Sophisticated Inference (Friston et al. 2020) and ToM extension (2508.00401v2):
- Discrete generative model with A/B/C/D matrices
- EFE = Pragmatic (preference satisfaction) + Epistemic (information gain about other's role)
- ToM: Q(a_other) proportional to softmax(-gamma * G_other), not greedy-x
- Collision avoidance via preferences C, not hard clamps
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

# Corridor geometry
X_MIN, X_MAX = -2.2, 2.2
Y_MIN, Y_MAX = -0.5, 0.5
ROBOT_RADIUS = 0.30
COLLISION_DIST = 2 * ROBOT_RADIUS  # 0.6m
CAUTION_DIST = 1.0  # meters

# X: 11 bins at 0.4m resolution
N_X = 11
X_EDGES = np.linspace(X_MIN, X_MAX, N_X + 1)
X_CENTERS = 0.5 * (X_EDGES[:-1] + X_EDGES[1:])

# Y: 5 bins for finer lateral resolution (enables lateral passing)
N_Y = 5
Y_EDGES = np.linspace(Y_MIN, Y_MAX, N_Y + 1)
Y_CENTERS = 0.5 * (Y_EDGES[:-1] + Y_EDGES[1:])

# Total pose states per agent
N_POSE = N_X * N_Y  # 33

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


# ============================================================================
# CorridorModel: Discrete generative model for the TIAGo corridor
# ============================================================================

class CorridorModel:
    """
    Discrete POMDP generative model for corridor navigation.

    State factors:
      - s_self:  33 pose states (11x * 3y)
      - s_other: 33 pose states
      - s_role:  4 hidden role states {PUSH, YIELD_LEFT, YIELD_RIGHT, WAIT}

    Observation modalities:
      - o_self:   33 (identity from s_self)
      - o_other:  33 (identity from s_other)
      - o_risk:   3  {safe, caution, danger} from joint pose geometry
      - o_motion: 4  {forward, backward, lateral, still} from s_role
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
        self.C_risk = np.array([0.0, 0.0, -6.0])
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
        """Preferences over self-position: goal attraction + distance gradient."""
        C = np.zeros(N_POSE)
        gx, gy = xy_to_bin(goal_x, goal_y)
        goal_flat = pose_to_flat(gx, gy)

        for s in range(N_POSE):
            x, y = flat_to_pose(s)
            if s == goal_flat:
                C[s] = 80.0
            else:
                manhattan = abs(x - gx) + abs(y - gy)
                C[s] = -3.0 * manhattan - 0.1
        return C


# ============================================================================
# EFE computation (JAX)
# ============================================================================

def _entropy(q):
    """Shannon entropy H(q) = -sum(q * log(q))."""
    return -jnp.sum(q * jnp.log(q + 1e-16))


def _compute_one_step_efe(
    q_self, q_other, q_role, action,
    B_self, B_other_pose, B_role,
    A_risk, A_motion,
    C_self, C_risk,
    epistemic_scale,
):
    """
    Compute single-step EFE for one action.

    Returns (G_step, q_self_next, q_other_next, q_role_next).

    EFE = G_pragmatic + G_epistemic

    G_pragmatic = -(C_self . q(s_self'))               [location utility]
                  -(C_risk . P(o_risk | s_self', s_other'))  [clearance utility]

    G_epistemic = -scale * [H(q(role')) - E_o[H(q(role'|o))]]  [info gain about role]
    """
    # 1. Transition beliefs
    q_self_next = B_self[:, :, action] @ q_self
    # Other: marginalize over role
    # q_other_next = sum_role q_role[role] * B_other_pose[:, :, role] @ q_other
    q_other_next = jnp.einsum('ij,j,k->i',
                               jnp.einsum('ijk,k->ij', B_other_pose, q_role),
                               q_other, jnp.ones(1))[..., None].squeeze(-1)
    # More explicitly:
    q_other_next = jnp.zeros(N_POSE)
    for r in range(N_ROLES):
        q_other_next = q_other_next + q_role[r] * (B_other_pose[:, :, r] @ q_other)

    q_role_next = B_role @ q_role

    # 2. Pragmatic value
    # Location utility
    u_loc = jnp.dot(q_self_next, C_self)

    # Risk utility: P(o_risk | s_self', s_other') = einsum over joint
    obs_risk = jnp.einsum('oij,i,j->o', A_risk, q_self_next, q_other_next)
    u_risk = jnp.dot(obs_risk, C_risk)

    G_pragmatic = -(u_loc + u_risk)

    # 3. Epistemic value: info gain about role
    H_prior = _entropy(q_role_next)

    # Expected observation distribution
    obs_motion = A_motion @ q_role_next  # P(o_motion) = sum_role P(o|role) * q(role)

    # Posterior entropy for each possible observation
    def _posterior_entropy(o_idx):
        likelihood = A_motion[o_idx, :]
        post_unnorm = likelihood * q_role_next
        Z = jnp.sum(post_unnorm) + 1e-16
        post = post_unnorm / Z
        return _entropy(post)

    H_posteriors = jax.vmap(_posterior_entropy)(jnp.arange(N_MOTION))
    H_post_expected = jnp.dot(obs_motion, H_posteriors)
    info_gain = H_prior - H_post_expected

    G_epistemic = -epistemic_scale * info_gain

    return G_pragmatic + G_epistemic, q_self_next, q_other_next, q_role_next


# We can't easily vmap the role loop above, so let's rewrite the other transition
# as a proper matrix multiply for JAX compatibility.
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
    """JAX-compatible single-step EFE. Returns (G_step, q_self', q_other', q_role')."""
    # Transitions
    q_self_next = B_self[:, :, action] @ q_self
    q_other_next = _transition_other_marginalized(q_other, q_role, B_other_pose)
    q_role_next = B_role @ q_role

    # Pragmatic: location
    u_loc = jnp.dot(q_self_next, C_self)

    # Pragmatic: risk
    obs_risk = jnp.einsum('oij,i,j->o', A_risk, q_self_next, q_other_next)
    u_risk = jnp.dot(obs_risk, C_risk)

    G_prag = -(u_loc + u_risk)

    # Epistemic: info gain about role
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
# ToM: Predict other agent's action
# ============================================================================

def predict_other_action_tom(
    q_other, q_self, other_model, other_alpha, gamma=8.0,
):
    """
    Theory of Mind: predict other's action distribution.

    Q(a_other) proportional to softmax(-gamma * G_other(a_other))

    G_other is computed from the other's perspective using their goal/preferences.
    """
    # Build other's EFE for each of their 5 actions
    G_others = np.zeros(N_ACTIONS)

    # From other's perspective: they are "self", we are "other"
    # Use a uniform role prior for their prediction of us
    uniform_role = np.ones(N_ROLES) / N_ROLES

    for a in range(N_ACTIONS):
        # Other's next state given their action a
        q_other_next = other_model.B_self[:, :, a] @ q_other

        # Location utility (from their preferences)
        u_loc = np.dot(q_other_next, other_model.C_self)

        # Risk utility (from their perspective)
        obs_risk = np.einsum('oij,i,j->o', other_model.A_risk, q_other_next, q_self)
        u_risk = np.dot(obs_risk, other_model.C_risk)

        G_others[a] = -(u_loc + u_risk)

    # Softmax action selection
    log_q = -gamma * G_others
    log_q -= log_q.max()
    q_a = np.exp(log_q)
    q_a /= q_a.sum()

    return q_a


def compute_other_best_response_G(
    q_self_next, q_other, other_model, other_alpha, depth=2,
):
    """
    Compute the other agent's best-response EFE given our predicted next position.

    Uses multi-step lookahead: tries all action sequences of given depth,
    returns min total G — the best EFE the other can achieve.
    """
    best_G = float('inf')
    for policy in itertools.product(range(N_ACTIONS), repeat=depth):
        q_o = q_other.copy()
        total_G = 0.0
        discount = 1.0
        for a in policy:
            q_o_next = other_model.B_self[:, :, a] @ q_o
            u_loc = np.dot(q_o_next, other_model.C_self)
            obs_risk = np.einsum('oij,i,j->o', other_model.A_risk, q_o_next, q_self_next)
            u_risk = np.dot(obs_risk, other_model.C_risk)
            total_G += discount * (-(u_loc + u_risk))
            discount *= 0.95
            q_o = q_o_next
        if total_G < best_G:
            best_G = total_G
    return best_G


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


def update_role_belief(q_role, motion_obs_idx, A_motion):
    """Bayesian update: q(role | o) proportional to P(o | role) * q(role)."""
    likelihood = A_motion[motion_obs_idx, :]
    posterior_unnorm = likelihood * q_role
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
        self.depth = 3              # Sophisticated inference depth
        self.gamma = 8.0            # Inverse temperature for softmax action selection
        self.epistemic_scale = 1.0  # Weight on epistemic term
        self.discount = 0.95        # Future EFE discount
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

        # JIT-compiled rollout (built lazily on first use)
        self._jit_rollout_all = None

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
            self.q_role = update_role_belief(
                self.q_role, motion_obs, self.model.A_motion
            )
            # Propagate role forward in time
            self.q_role = self.model.B_role @ self.q_role

        self.prev_other_x = other_x
        self.prev_other_y = other_y

        # 2. Discretize current positions into beliefs
        q_self = make_delta_belief(my_x, my_y)
        q_other = make_delta_belief(other_x, other_y)

        # 3. Compute self EFE for all policies
        G_self_all = self._compute_all_efe(q_self, q_other, self.q_role)

        # 4. Compute social EFE term
        other_model = self._get_other_model(other_goal_x, other_goal_y)

        # For each first action, compute G_other (other's best response)
        G_social_all = np.copy(G_self_all)
        if self.alpha > 0.001:
            # Compute other's best-response EFE for each of our first actions
            G_other_per_first_action = np.zeros(N_ACTIONS)
            for a in range(N_ACTIONS):
                # Our predicted next state
                q_self_next_a = self.model.B_self[:, :, a] @ q_self
                G_other_per_first_action[a] = compute_other_best_response_G(
                    q_self_next_a, q_other, other_model, other_alpha
                )

            # Add alpha-weighted other EFE to each policy based on its first action
            for i, policy in enumerate(self.all_policies):
                first_a = policy[0]
                G_social_all[i] += self.alpha * G_other_per_first_action[first_a]

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
    for x in [-2.2, -1.0, 0.0, 1.0, 2.2]:
        for y in [-0.5, 0.0, 0.5]:
            bx, by = xy_to_bin(x, y)
            cx, cy = bin_to_xy(bx, by)
            assert abs(cx - x) < 0.4, f"X round-trip failed: {x} -> bin {bx} -> {cx}"
            assert abs(cy - y) < 0.25, f"Y round-trip failed: {y} -> bin {by} -> {cy}"
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
    test_efe_basic()
    print()
    test_planner()
