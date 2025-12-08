"""
TOM-style active inference planners for LavaCorridor.

This package contains progressive implementations of active inference planners:
- Phase 1: Single-agent EFE-only planner (si_lava.py)
- Phase 2: Empathic multi-agent planner (si_empathy_lava.py)
- Phase 3: F-aware planner (si_F_aware_lava.py) - coming soon
"""

from .si_lava import (
    propagate_state,
    compute_risk_G,
    efe_risk_only,
    LavaPlanner,
)

from .si_empathy_lava import (
    compute_other_agent_G,
    compute_empathic_G,
    efe_empathic,
    EmpathicLavaPlanner,
)

__all__ = [
    # Phase 1: Single-agent
    "propagate_state",
    "compute_risk_G",
    "efe_risk_only",
    "LavaPlanner",
    # Phase 2: Empathic
    "compute_other_agent_G",
    "compute_empathic_G",
    "efe_empathic",
    "EmpathicLavaPlanner",
]
