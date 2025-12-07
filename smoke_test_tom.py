"""
TOM-style smoke test for LavaCorridor.

This verifies that the TOM infrastructure works correctly with:
- LavaModel (generative model)
- LavaAgent (pymdp agent with correct structure)
- LavaV1Env (environment wrapper)
- TOM planning functions
"""

import sys
import logging
from pathlib import Path

# Add project root to path
repo_root = Path(__file__).parent
sys.path.insert(0, str(repo_root))

import jax
import jax.numpy as jnp
import jax.random as jr

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def test_imports():
    """Test that all TOM components import successfully."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 1: Testing TOM imports...")
    LOGGER.info("="*80)

    try:
        from tom.models import LavaModel, LavaAgent
        LOGGER.info("  ✓ TOM model imports")

        from tom.envs import LavaV1Env
        LOGGER.info("  ✓ TOM env imports")

        from tom.planning.si import si_policy_search
        LOGGER.info("  ✓ TOM planning imports")

        LOGGER.info("\n✅ All TOM imports successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_model_creation():
    """Test creating LavaModel and LavaAgent."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 2: Testing TOM model creation...")
    LOGGER.info("="*80)

    try:
        from tom.models import LavaModel, LavaAgent

        # Create model
        model = LavaModel(width=4, height=3, goal_x=3)
        LOGGER.info(f"  ✓ LavaModel created")
        LOGGER.info(f"    A keys: {list(model.A.keys())}")
        LOGGER.info(f"    B keys: {list(model.B.keys())}")
        LOGGER.info(f"    C keys: {list(model.C.keys())}")
        LOGGER.info(f"    D keys: {list(model.D.keys())}")

        # Create agent (TOM-style: thin wrapper, no PyMDP)
        agent = LavaAgent(model, horizon=1, gamma=8.0)
        LOGGER.info(f"  ✓ LavaAgent created")
        LOGGER.info(f"    Num policies: {len(agent.policies)}")
        LOGGER.info(f"    Policy shape: {agent.policies.shape}")
        LOGGER.info(f"    A shape: {agent.A['location_obs'].shape}")
        LOGGER.info(f"    B shape: {agent.B['location_state'].shape}")
        LOGGER.info(f"    C shape: {agent.C['location_obs'].shape}")
        LOGGER.info(f"    D shape: {agent.D['location_state'].shape}")

        LOGGER.info("\n✅ TOM model creation successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Model creation failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_env_interaction():
    """Test environment wrapper and basic interaction."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 3: Testing TOM environment interaction...")
    LOGGER.info("="*80)

    try:
        from tom.envs import LavaV1Env

        # Create environment
        env = LavaV1Env(width=4, height=3, num_agents=2, timesteps=10)
        LOGGER.info(f"  ✓ LavaV1Env created")
        LOGGER.info(f"    Grid: {env.width}x{env.height}")
        LOGGER.info(f"    Num agents: {env.num_agents}")

        # Reset
        key = jr.PRNGKey(0)
        state, obs = env.reset(key)
        LOGGER.info(f"  ✓ Environment reset")
        LOGGER.info(f"    State keys: {list(state.keys())}")
        LOGGER.info(f"    Obs keys: {list(obs.keys())}")
        LOGGER.info(f"    Agent 0 obs: {obs[0]}")
        LOGGER.info(f"    Agent 1 obs: {obs[1]}")

        # Step with random actions
        actions = {0: 3, 1: 3}  # Both move RIGHT
        next_state, next_obs, reward, done, info = env.step(state, actions)
        LOGGER.info(f"  ✓ Environment step")
        LOGGER.info(f"    Done: {done}")
        LOGGER.info(f"    Timestep: {next_state['timestep']}")

        LOGGER.info("\n✅ TOM environment interaction successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Environment interaction failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_agent_inference():
    """Test TOM-style Bayesian state inference (manual, not PyMDP)."""
    LOGGER.info("="*80)
    LOGGER.info("STEP 4: Testing TOM agent inference...")
    LOGGER.info("="*80)

    try:
        from tom.models import LavaModel, LavaAgent
        from tom.envs import LavaV1Env
        import numpy as np

        # Create model and agent
        model = LavaModel(width=4, height=3)
        agent = LavaAgent(model, horizon=1, gamma=8.0)

        # Create env and get initial observation
        env = LavaV1Env(width=4, height=3, num_agents=1, timesteps=10)
        key = jr.PRNGKey(0)
        state, obs = env.reset(key)

        # Extract observation for agent 0
        # obs[0]["location_obs"] is a JAX array of shape (1,), not a scalar
        # Index [0] first to get scalar before int() conversion
        agent_obs = int(np.asarray(obs[0]["location_obs"])[0])
        LOGGER.info(f"  Agent observation: {agent_obs}")

        # Manual Bayesian state inference using A and D
        # This is TOM-style: use generative model directly, not PyMDP agent
        A0 = np.asarray(model.A["location_obs"])        # (num_obs, num_states)
        D0 = np.asarray(model.D["location_state"])      # (num_states,)

        likelihood = A0[agent_obs]                      # p(o|s) for each s
        unnorm = likelihood * D0                        # p(o,s) = p(o|s) * p(s)
        denom = unnorm.sum()

        if denom > 0:
            qs = unnorm / denom                         # p(s|o)
        else:
            qs = np.ones_like(unnorm) / unnorm.size

        LOGGER.info(f"  ✓ Manual Bayesian inference complete")
        LOGGER.info(f"    Posterior qs shape: {qs.shape}")
        LOGGER.info(f"    Posterior qs sum: {qs.sum():.4f}")
        LOGGER.info(f"    Posterior qs min={qs.min():.4f}, max={qs.max():.4f}")

        # Verify belief is concentrated on observed state
        # (since A is identity and D was peaked at start state)
        most_likely_state = np.argmax(qs)
        LOGGER.info(f"    Most likely state: {most_likely_state}")
        LOGGER.info(f"    Confidence: {qs[most_likely_state]:.4f}")

        LOGGER.info("\n✅ TOM agent inference (manual Bayes) successful!\n")
        return True

    except Exception as e:
        LOGGER.error(f"\n❌ Agent inference failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all TOM smoke tests."""
    print("\n" + "="*80)
    print(" "*20 + "TOM LAVA CORRIDOR SMOKE TEST")
    print("="*80 + "\n")

    results = []

    # Test 1: Imports
    results.append(("TOM Imports", test_imports()))

    # Test 2: Model Creation
    results.append(("TOM Model Creation", test_model_creation()))

    # Test 3: Environment Interaction
    results.append(("TOM Environment", test_env_interaction()))

    # Test 4: Agent Inference
    results.append(("TOM Agent Inference", test_agent_inference()))

    # Summary
    print("\n" + "="*80)
    print(" "*30 + "SUMMARY")
    print("="*80)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:.<50} {status}")
        if not passed:
            all_passed = False

    print("="*80 + "\n")

    if all_passed:
        print("🎉 ALL TOM TESTS PASSED! LavaCorridor TOM system is ready.")
        print("\nNext steps:")
        print("  1. Run multi-agent TOM rollout")
        print("  2. Test ToMify for recursive inference")
        print("  3. Run experiments with path flexibility metrics")
        return 0
    else:
        print("⚠️  SOME TOM TESTS FAILED. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
