# TOM-Compatible Test Suite

This directory contains tests for the **TOM-style pure JAX architecture**.

## Overview

The TOM-compatible tests verify the new architecture that uses:
- **LavaModel**: Pure JAX dataclass with dict-structured A, B, C, D
- **LavaAgent**: Thin wrapper around model (no PyMDP Agent inheritance)
- **LavaV1Env**: JAX environment with multi-agent support
- **Manual Bayesian inference**: Explicit state updates (no PyMDP `infer_states`)

## Test Files

### ✅ TOM-Compatible Tests (Active)

#### 1. `smoke_test_tom.py`
**Location**: Repository root
**Purpose**: Quick verification that TOM infrastructure works
**Tests**:
- TOM imports (LavaModel, LavaAgent, LavaV1Env)
- Model creation with dict-structured A, B, C, D
- Environment interaction
- Manual Bayesian inference

**Run**:
```bash
python smoke_test_tom.py
```

#### 2. `test_lava_env_tom.py`
**Purpose**: Test TOM-style environment and model
**Tests**:
- LavaModel creation and structure
- LavaAgent creation and policies
- LavaV1Env reset and step
- Transition dynamics (B matrix)
- Preference structure (C vector)
- Initial state prior (D vector)
- Collision detection

**Run**:
```bash
pytest tests/test_lava_env_tom.py -v
```

#### 3. `test_model_creation_tom.py`
**Purpose**: Comprehensive model and agent creation tests
**Tests**:
- Model creation (basic, custom goal, different sizes)
- Dict structure (A, B, C, D as dicts, not lists)
- Matrix shapes (A, B, C, D dimensions)
- Matrix properties (identity A, stochastic B, valid probs)
- Transition dynamics (STAY, RIGHT, UP, boundaries)
- Agent creation and policy structure

**Run**:
```bash
pytest tests/test_model_creation_tom.py -v
```

#### 4. `test_integration_tom.py`
**Purpose**: Integration tests for TOM components working together
**Tests**:
- Creating all components together
- Model-env compatibility
- Manual Bayesian inference from env observations
- Belief updates after actions
- Policy forward simulation
- Multi-agent interactions
- End-to-end scenario (observe → infer → predict → act)

**Run**:
```bash
pytest tests/test_integration_tom.py -v
```

#### 5. `test_path_flexibility_metrics.py`
**Purpose**: Test path flexibility metrics (E, R, O, F)
**Tests**:
- Empowerment computation
- Returnability computation
- Overlap computation
- Combined flexibility metric
- Edge cases and numerical stability

**Run**:
```bash
pytest tests/test_path_flexibility_metrics.py -v
```

#### 6. `test_F_aware_prior.py`
**Purpose**: Test F-aware policy prior
**Tests**:
- κ=0 recovers baseline
- κ>0 biases toward high-F policies
- β weighting (individual vs joint flexibility)
- EFE-flexibility tradeoff
- Numerical stability

**Run**:
```bash
pytest tests/test_F_aware_prior.py -v
```

## Running All Tests

Use the provided test runner:

```bash
python run_all_tests.py
```

This runs all TOM-compatible tests in the correct order and provides a summary.

## Test Structure

```
tests/
├── smoke_test_tom.py              # Quick TOM smoke test
├── test_lava_env_tom.py           # TOM environment tests
├── test_model_creation_tom.py     # TOM model/agent creation
├── test_integration_tom.py        # TOM integration tests
├── test_path_flexibility_metrics.py  # Path flexibility (E, R, O, F)
└── test_F_aware_prior.py          # F-aware policy prior
```

## Legacy PyMDP Tests (Deprecated)

The following tests are **not compatible** with TOM-style architecture and have been replaced:

| Legacy Test | Status | TOM Replacement |
|-------------|--------|-----------------|
| `smoke_test.py` | ❌ Always fails (PyMDP) | `smoke_test_tom.py` |
| `test_lava_rollout.py` | ❌ Uses PyMDP LavaCorridorEnv | `test_lava_env_tom.py` |
| `test_agent_factory.py` | ❌ Uses PyMDP agents | `test_model_creation_tom.py` |
| `test_integration_rollout.py` | ❌ Uses PyMDP rollouts | `test_integration_tom.py` |

## What Gets Tested

### Core TOM Components
- ✅ Pure JAX generative models (LavaModel)
- ✅ Thin agent wrappers (LavaAgent)
- ✅ JAX environments (LavaV1Env)
- ✅ Manual Bayesian inference
- ✅ Dict-structured A, B, C, D (not lists)

### Functionality
- ✅ Model-environment compatibility
- ✅ State inference from observations
- ✅ Belief updates using B matrix
- ✅ Policy forward simulation
- ✅ Multi-agent coordination
- ✅ Path flexibility metrics (E, R, O, F)
- ✅ F-aware policy prior

### Edge Cases
- ✅ Different grid sizes
- ✅ Different goal positions
- ✅ Boundary handling
- ✅ Collision detection
- ✅ Numerical stability

## Test Coverage Summary

| Component | Unit Tests | Integration Tests | Smoke Tests |
|-----------|-----------|-------------------|-------------|
| LavaModel | ✅ | ✅ | ✅ |
| LavaAgent | ✅ | ✅ | ✅ |
| LavaV1Env | ✅ | ✅ | ✅ |
| Manual Inference | ✅ | ✅ | ✅ |
| Path Flexibility | ✅ | ❌ | ❌ |
| F-Aware Prior | ✅ | ❌ | ❌ |

## Expected Output

When all tests pass:

```
================================================================================
                            SUMMARY
================================================================================
ALL TOM-COMPATIBLE TESTS PASSED! 🎉

TOM-style JAX architecture is verified and working.

================================================================================
What's been tested:
  ✓ LavaModel (pure JAX generative model with dict structure)
  ✓ LavaAgent (thin wrapper around model)
  ✓ LavaV1Env (JAX environment with multi-agent support)
  ✓ Manual Bayesian inference (no PyMDP)
  ✓ Policy evaluation using B matrix
  ✓ Path flexibility metrics (E, R, O, F)
  ✓ F-aware policy prior
  ✓ Multi-agent interactions

================================================================================
Next steps to complete TOM integration:
  1. Add TOM-style EFE computation (port from tom/planning/si_tom.py)
  2. Implement policy search using computed EFE
  3. Add multi-agent TOM rollouts (agents reasoning about each other)
  4. Connect to path flexibility metrics during planning
  5. Run Experiments 1 & 2 with full TOM pipeline
================================================================================
```

## Troubleshooting

### Test Failures

If tests fail, check:
1. **Import errors**: Ensure `tom/models/__init__.py` exports `LavaModel` and `LavaAgent`
2. **Import errors**: Ensure `tom/envs/__init__.py` exports `LavaV1Env`
3. **Shape mismatches**: Verify A, B, C, D are dicts (not lists)
4. **JAX errors**: Ensure JAX arrays are converted with `np.asarray()` before indexing

### Common Issues

**Issue**: `TypeError: list indices must be integers or slices, not str`
**Solution**: Check that A, B, C, D are dicts (`model.A = {"key": array}`) not lists (`model.A = [array]`)

**Issue**: `IndexError: axis 1 is out of bounds for array of dimension 1`
**Solution**: Use `int(np.asarray(obs[0]["location_obs"])[0])` to extract scalar from JAX array

**Issue**: `Only scalar arrays can be converted to Python scalars`
**Solution**: Index JAX array first: `arr[0]` before calling `int()`

## Next Steps

After all tests pass:
1. Port TOM EFE computation from `tom/planning/si_tom.py`
2. Implement policy search using EFE
3. Add multi-agent TOM rollouts
4. Connect path flexibility metrics to planning
5. Run full experiments

---

**Note**: This test suite is part of the transition to TOM-style pure JAX architecture. Legacy PyMDP tests are deprecated and should not be used.
