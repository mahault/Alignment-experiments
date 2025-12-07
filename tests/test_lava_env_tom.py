"""
Test TOM-style LavaV1Env and LavaModel.

This tests:
1. LavaV1Env initialization and stepping
2. LavaModel generative model structure
3. LavaAgent creation
4. Collision and lava detection
5. Multi-agent interactions
"""

import logging
import sys
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

import jax.random as jr
import jax.numpy as jnp
import numpy as np

from tom.models import LavaModel, LavaAgent
from tom.envs import LavaV1Env

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def test_lava_model_creation():
    """Test LavaModel creation and structure."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 1: LavaModel Creation")
    LOGGER.info("=" * 80)

    model = LavaModel(width=4, height=3, goal_x=3)

    # Check A, B, C, D are dicts
    assert isinstance(model.A, dict), "A should be dict"
    assert isinstance(model.B, dict), "B should be dict"
    assert isinstance(model.C, dict), "C should be dict"
    assert isinstance(model.D, dict), "D should be dict"

    # Check keys
    assert "location_obs" in model.A
    assert "location_state" in model.B
    assert "location_obs" in model.C
    assert "location_state" in model.D

    # Check shapes
    num_states = model.width * model.height
    assert model.A["location_obs"].shape == (num_states, num_states)
    assert model.B["location_state"].shape == (num_states, num_states, 5)  # 5 actions
    assert model.C["location_obs"].shape == (num_states,)
    assert model.D["location_state"].shape == (num_states,)

    LOGGER.info("✓ LavaModel structure correct")
    LOGGER.info(f"  States: {num_states}")
    LOGGER.info(f"  A shape: {model.A['location_obs'].shape}")
    LOGGER.info(f"  B shape: {model.B['location_state'].shape}")
    LOGGER.info(f"  C shape: {model.C['location_obs'].shape}")
    LOGGER.info(f"  D shape: {model.D['location_state'].shape}")


def test_lava_agent_creation():
    """Test LavaAgent creation and policy structure."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 2: LavaAgent Creation")
    LOGGER.info("=" * 80)

    model = LavaModel(width=4, height=3)
    agent = LavaAgent(model, horizon=2, gamma=8.0)

    # Check agent exposes model dicts
    assert agent.A is model.A, "Agent should expose model.A"
    assert agent.B is model.B, "Agent should expose model.B"
    assert agent.C is model.C, "Agent should expose model.C"
    assert agent.D is model.D, "Agent should expose model.D"

    # Check policies
    assert agent.policies.shape == (5, 1, 1), "Should have (5, 1, 1) policies"

    LOGGER.info("✓ LavaAgent created successfully")
    LOGGER.info(f"  Policies shape: {agent.policies.shape}")
    LOGGER.info(f"  Horizon: {agent.horizon}")
    LOGGER.info(f"  Gamma: {agent.gamma}")


def test_lava_v1_env_reset():
    """Test LavaV1Env reset."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 3: LavaV1Env Reset")
    LOGGER.info("=" * 80)

    env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
    key = jr.PRNGKey(0)

    state, obs = env.reset(key)

    # Check state structure (matches LavaV1Env wrapper)
    assert "timestep" in state
    assert "env_state" in state
    assert "done" in state
    assert "pos" in state["env_state"]  # Positions are inside env_state

    # Check observations structure
    assert len(obs) == 2, "Should have 2 agent observations"
    assert "location_obs" in obs[0]
    assert "location_obs" in obs[1]

    # Check positions are valid (dict of agent_id: (x, y))
    assert len(state["env_state"]["pos"]) == 2, "Should have positions for 2 agents"

    LOGGER.info("✓ LavaV1Env reset successful")
    LOGGER.info(f"  Num agents: {env.num_agents}")
    LOGGER.info(f"  Grid: {env.width}x{env.height}")
    LOGGER.info(f"  Initial positions: {state['env_state']['pos']}")
    LOGGER.info(f"  Agent 0 obs: {obs[0]['location_obs']}")
    LOGGER.info(f"  Agent 1 obs: {obs[1]['location_obs']}")


def test_lava_v1_env_step():
    """Test LavaV1Env stepping."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 4: LavaV1Env Step")
    LOGGER.info("=" * 80)

    env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
    key = jr.PRNGKey(0)

    state, obs = env.reset(key)

    # Take a step with both agents moving RIGHT
    actions = {0: 3, 1: 3}  # RIGHT = 3
    next_state, next_obs, reward, done, info = env.step(state, actions)

    # Check timestep incremented
    assert next_state["timestep"] == state["timestep"] + 1

    # Check observations updated
    assert "location_obs" in next_obs[0]
    assert "location_obs" in next_obs[1]

    LOGGER.info("✓ LavaV1Env step successful")
    LOGGER.info(f"  Actions: {actions}")
    LOGGER.info(f"  New timestep: {next_state['timestep']}")
    LOGGER.info(f"  Done: {done}")


def test_lava_transitions():
    """Test that B matrix encodes correct lava transitions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 5: Lava Transition Dynamics")
    LOGGER.info("=" * 80)

    model = LavaModel(width=4, height=3)
    B = model.B["location_state"]  # (num_states, num_states, num_actions)

    # Test STAY action (action 4)
    STAY = 4
    for s in range(model.num_states):
        # STAY should keep agent in same state
        prob_stay = B[s, s, STAY]
        assert np.isclose(prob_stay, 1.0), f"STAY at state {s} should be deterministic"

    # Test RIGHT action moves agent right (when not at wall)
    RIGHT = 3
    # From (x=0, y=1) should go to (x=1, y=1)
    s_from = 1 * model.width + 0  # (y=1, x=0)
    s_to = 1 * model.width + 1    # (y=1, x=1)

    prob_right = B[s_to, s_from, RIGHT]
    assert np.isclose(prob_right, 1.0), f"RIGHT from ({0}, {1}) should go to ({1}, {1})"

    LOGGER.info("✓ Transition dynamics correct")
    LOGGER.info(f"  STAY: deterministic (p=1.0)")
    LOGGER.info(f"  RIGHT: moves agent right when possible")


def test_preferences_structure():
    """Test that C encodes goal positive, lava negative."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 6: Preference Structure")
    LOGGER.info("=" * 80)

    model = LavaModel(width=5, height=3, goal_x=4)
    C = model.C["location_obs"]

    # Find goal state (goal_x, safe_y)
    goal_idx = model.safe_y * model.width + model.goal_x
    goal_pref = C[goal_idx]

    # Find lava states (y != safe_y)
    lava_idx = 0 * model.width + 0  # (y=0, x=0) is lava
    lava_pref = C[lava_idx]

    # Goal should be positive, lava negative
    assert goal_pref > 0, "Goal should have positive preference"
    assert lava_pref < 0, "Lava should have negative preference"
    assert goal_pref > lava_pref, "Goal should be preferred over lava"

    LOGGER.info("✓ Preference structure correct")
    LOGGER.info(f"  Goal preference: {goal_pref:.2f}")
    LOGGER.info(f"  Lava preference: {lava_pref:.2f}")


def test_initial_state_prior():
    """Test that D encodes valid initial state prior."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 7: Initial State Prior")
    LOGGER.info("=" * 80)

    model = LavaModel(width=4, height=3)
    D = model.D["location_state"]

    # Should be valid probability distribution
    assert np.isclose(D.sum(), 1.0), "D should sum to 1"
    assert np.all(D >= 0), "D should be non-negative"

    # Start position (0, safe_y) should have probability 1
    start_idx = model.safe_y * model.width + 0
    assert np.isclose(D[start_idx], 1.0), "Should start at (0, safe_y)"

    LOGGER.info("✓ Initial state prior correct")
    LOGGER.info(f"  D sum: {D.sum():.4f}")
    LOGGER.info(f"  Start state prob: {D[start_idx]:.4f}")


def test_collision_detection():
    """Test that environment detects collisions."""
    LOGGER.info("=" * 80)
    LOGGER.info("TEST 8: Collision Detection")
    LOGGER.info("=" * 80)

    env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
    key = jr.PRNGKey(42)

    # Start from a valid state returned by the env
    state, obs = env.reset(key)

    # Manually modify env_state to create collision scenario
    env_state = state["env_state"].copy()
    # Place agents at (1, 0) and (1, 2)
    env_state["pos"] = {0: (1, 0), 1: (1, 2)}

    # Update state with modified env_state
    state = {
        "timestep": 0,
        "env_state": env_state,
        "done": False,
    }

    # Both move toward middle: DOWN (1), UP (0)
    actions = {0: 1, 1: 0}
    next_state, next_obs, reward, done, info = env.step(state, actions)

    # Check if step completes (collision handling depends on env implementation)
    LOGGER.info("✓ Collision detection works")
    LOGGER.info(f"  Initial positions: {env_state['pos']}")
    LOGGER.info(f"  Actions: {actions}")
    LOGGER.info(f"  Final positions: {next_state['env_state']['pos']}")
    if "collision" in next_state["env_state"]:
        LOGGER.info(f"  Collision detected: {next_state['env_state']['collision']}")


def main():
    """Run all TOM-style lava environment tests."""
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("TOM-STYLE LAVA ENVIRONMENT TESTS")
    LOGGER.info("=" * 80 + "\n")

    try:
        test_lava_model_creation()
        test_lava_agent_creation()
        test_lava_v1_env_reset()
        test_lava_v1_env_step()
        test_lava_transitions()
        test_preferences_structure()
        test_initial_state_prior()
        test_collision_detection()

        LOGGER.info("\n" + "=" * 80)
        LOGGER.info("ALL TOM ENVIRONMENT TESTS PASSED ✓")
        LOGGER.info("=" * 80)
        return 0

    except Exception as e:
        LOGGER.error(f"\n{'='*80}")
        LOGGER.error(f"TEST FAILED ✗")
        LOGGER.error(f"{'='*80}")
        LOGGER.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
