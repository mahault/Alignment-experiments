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

## Legacy Code

Old controllers and world files from earlier iterations are preserved in `../legacy/webots/`.
