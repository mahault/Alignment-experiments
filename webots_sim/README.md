# Webots Integration for Alignment Experiments

This directory contains a Webots implementation of the multi-agent empathic coordination experiments.

## Overview

The Webots simulation mirrors the discrete grid environments from the alignment-experiments framework, but with physical robot dynamics. Two differential-drive robots navigate toward each other's starting positions while using Active Inference planning with Theory of Mind and empathy.

## Structure

```
webots_sim/
├── controllers/
│   └── empathic_navigator/
│       └── empathic_navigator.py   # Robot controller with Active Inference
├── protos/
│   ├── WebotsAcrome.proto          # Differential-drive robot with sensors
│   ├── HazardObstacle.proto        # Lava/hazard obstacles
│   └── Target.proto                # Goal markers
├── worlds/
│   ├── narrow_corridor.wbt         # 6x3 narrow passage (hard)
│   └── bottleneck.wbt              # 8x3 with central bottleneck
├── run_webots_experiment.py        # Experiment launcher
└── README.md
```

## Requirements

- Webots R2025a (installed in `../webots/`)
- Python 3.8+
- alignment-experiments dependencies (numpy, jax)

## Quick Start

1. **Open a world directly in Webots:**
   ```bash
   ../webots/msys64/mingw64/bin/webots.exe worlds/narrow_corridor.wbt
   ```

2. **Run via script:**
   ```bash
   python run_webots_experiment.py --world narrow_corridor.wbt --alpha 0.5
   ```

3. **Run parameter sweep:**
   ```bash
   python run_webots_experiment.py --world narrow_corridor.wbt --sweep
   ```

## Robot Configuration

Robots are configured via the `customData` field in the world file:
```
customData "goal_x,goal_y,alpha,grid_width,grid_height"
```

Example: `"1.25,0.25,0.5,6,3"` means:
- Goal position: (1.25, 0.25)
- Empathy parameter α = 0.5
- Grid: 6×3 cells

## Empathy Parameter (α)

The empathy parameter controls how much each agent weighs the other's utility:
- `α = 0.0`: Pure self-interest (greedy)
- `α = 0.5`: Balanced consideration (default)
- `α = 1.0`: Full empathy (other's utility weighted equally)

Higher empathy can lead to:
- Better coordination in narrow passages
- Potential "paralysis" if both agents defer too much

## Grid Mapping

The Webots continuous space maps to the discrete grid used by Active Inference:

| Grid Cell | Webots Position (m) |
|-----------|---------------------|
| (0, 0)    | (-1.25, -0.25)      |
| (0, 1)    | (-1.25, 0.25)       |
| (5, 1)    | (1.25, 0.25)        |

Cell size: 0.5m × 0.5m

## Communication

Robots communicate their grid positions via Webots Emitter/Receiver devices. This enables each robot to know the other's location for Theory of Mind reasoning.

## Integrating with Alignment Experiments

The controller imports planning modules from the parent alignment-experiments:

```python
from tom.models.model_lava import LavaModel, LavaAgent
from tom.planning.si_empathy_lava import EmpathicLavaPlanner
```

To run with full Active Inference planning:
1. Ensure alignment-experiments dependencies are installed
2. Set `WEBOTS_LAYOUT` and `WEBOTS_START_CONFIG` environment variables
3. The controller will use empathic planning with recursive ToM

## Available Worlds

| World | Description | Coordination Challenge |
|-------|-------------|------------------------|
| `narrow_corridor.wbt` | 6×3 single-file passage | Must coordinate to pass |
| `bottleneck.wbt` | 8×3 with central squeeze | Sequential bottleneck |

## Extending

To add new layouts:
1. Create a new `.wbt` file in `worlds/`
2. Define floor, hazards, walls, targets, and robots
3. Match the grid layout to an existing `env_lava_variants.py` layout
4. Update robot `customData` with correct grid dimensions
