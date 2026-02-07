# Webots TIAGo Simulation

Physical robot simulation of empathic multi-agent coordination using Active Inference with Theory of Mind.

Two TIAGo robots navigate toward each other's starting positions in a corridor with hazard obstacles. The **empathic** robot (high alpha) yields to let the **selfish** robot (low alpha) pass — this behavior emerges from Expected Free Energy (EFE) computation, not hard-coded rules.

## Structure

```
webots_sim/
├── controllers/
│   └── tiago_empathic/
│       ├── tiago_empathic.py      # Robot controller (motor control, sensor reading)
│       ├── tom_planner.py         # ToM planner: discrete POMDP with EFE + social inference
│       └── tom_planner_legacy.py  # Previous version (deprecated, kept for reference)
├── protos/
│   ├── HazardObstacle.proto       # Lava/hazard obstacles
│   ├── Target.proto               # Goal markers
│   └── WebotsAcrome.proto         # Differential-drive robot base
├── worlds/
│   └── tiago_empathic_test.wbt   # Main world: two TIAGo robots in corridor
├── run_webots_experiment.py       # Experiment launcher script
├── ROADMAP.md                     # Technical design notes and fix history
└── README.md                      # This file
```

## Requirements

- [Webots R2025a](https://cyberbotics.com/) (installed in `../webots/`)
- Python 3.8+ with JAX, NumPy

## Quick Start

**Open directly in Webots:**
```bash
../webots/msys64/mingw64/bin/webots.exe worlds/tiago_empathic_test.wbt
```

**Or via the launcher script:**
```bash
python run_webots_experiment.py --world tiago_empathic_test.wbt --timeout 300
```

**Run the planner standalone (no Webots needed):**
```bash
cd controllers/tiago_empathic
python tom_planner.py
```
This runs unit tests and a two-robot simulation, showing the empathic robot yielding laterally.

## How It Works

### Planner (`tom_planner.py`)

A discrete POMDP generative model with proper Active Inference:

- **State space**: 77 pose states (11 X bins x 7 Y bins) covering the full 5x2m arena
- **Actions**: {STAY, FORWARD, BACK, LEFT, RIGHT}
- **EFE = Pragmatic + Epistemic**: goal attraction, hazard avoidance, info gain about other's role
- **Theory of Mind**: predicts other's best response via `Q(a_other) ~ softmax(-gamma * G_other)`
- **Social EFE**: `G_social = G_self + alpha * G_other` — alpha controls empathy
- **Sophisticated inference**: depth-5 rollout (3125 policies), JAX-parallelized
- **Hazard-aware**: arena geometry encoded in preferences — planner knows where open areas are

See [ROADMAP.md](ROADMAP.md) for detailed technical documentation.

### Controller (`tiago_empathic.py`)

Differential-drive controller with:
- Two-phase motor primitive (rotate-then-translate for decisive lateral moves)
- Backward driving when target is behind the robot
- Arm tucking to reduce collision footprint

### Robot Configuration

Robots are configured via `customData` in the world file:
```
customData "goal_x, goal_y, alpha, agent_id"
```

| Parameter | Description |
|-----------|-------------|
| `goal_x, goal_y` | Target position in meters |
| `alpha` | Empathy weight (0.0 = selfish, 6.0 = highly empathic) |
| `agent_id` | Unique ID (0 or 1) |

### Arena Geometry

The arena is 5m x 2m with hazard obstacles forming a narrow corridor:

```
Y=1.0  ┌─────────────────────────────────────────────┐
       │             ┌───┐ ┌───┐                      │  <- top hazards (X=-0.5, 0.5)
Y=0.5  │   OPEN      │ H │ │ H │      OPEN           │
       │             └───┘ └───┘                      │
Y=0.0  │  R1 ●─────────── corridor ───────────● R2   │
       │   ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐ ┌───┐     │  <- bottom hazards (continuous)
Y=-0.5 │   │ H │ │ H │ │ H │ │ H │ │ H │ │ H │     │
       │   └───┘ └───┘ └───┘ └───┘ └───┘ └───┘     │
Y=-1.0 └─────────────────────────────────────────────┘
      X=-2.5                  0                  X=2.5
```

Key insight: the **top side opens up** at |X| > 0.8 (no hazards). The empathic robot discovers this and yields into the open area.

## Available Worlds

All 11 layouts from `tom/envs/env_lava_variants.py` are available as Webots worlds, plus the original hand-tuned reference world. Generate them with:

```bash
python webots_sim/generate_worlds.py
```

| World File | Grid | Arena | Coordination Challenge |
|------------|------|-------|------------------------|
| `tiago_empathic_test.wbt` | - | 5x2m | Hand-tuned reference corridor |
| `tiago_narrow.wbt` | 6x3 | 5.2x3.1m | Single-file corridor, collision unavoidable |
| `tiago_wide.wbt` | 6x4 | 5.2x3.8m | Two-lane corridor, easy passing |
| `tiago_bottleneck.wbt` | 8x4 | 6.6x3.8m | Wide areas with narrow center choke point |
| `tiago_crossed_goals.wbt` | 6x4 | 5.2x3.8m | Agents must swap lanes to reach goals |
| `tiago_risk_reward.wbt` | 8x4 | 6.6x3.8m | Fast risky path vs slow safe detour |
| `tiago_double_bottleneck.wbt` | 10x4 | 8.0x3.8m | Two choke points with passing bay between |
| `tiago_passing_bay.wbt` | 8x4 | 6.6x3.8m | Narrow corridor with one 2-cell bay for yielding |
| `tiago_asymmetric_detour.wbt` | 8x4 | 6.6x3.8m | One agent has direct path, other must detour |
| `tiago_t_junction.wbt` | 7x5 | 5.9x4.5m | Agents approach from different directions |
| `tiago_symmetric_bottleneck.wbt` | 10x4 | 8.0x3.8m | Opposite sides through same bottleneck |
| `tiago_vertical_bottleneck.wbt` | 6x8 | 5.2x6.6m | Vertical corridor with wide passing areas |

### Auto-Discovery

The controller automatically discovers arena geometry at startup:
- Reads `RectangleArena.floorSize` to determine coordinate bounds
- Finds all `HazardObstacle` nodes and extracts their positions/sizes
- Calls `tom_planner.configure()` to set up the correct discretization

This means the same controller works for any world file without code changes.

## Legacy Code

Old controllers and world files from earlier iterations are preserved in `../legacy/webots/`.
