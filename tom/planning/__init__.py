"""
TOM-style active inference planners for LavaCorridor.

This package contains progressive implementations of active inference planners:
- Phase 1: Single-agent EFE-only planner (si_lava.py)
- Phase 2: Empathic multi-agent planner (si_empathy_lava.py)
- Phase 3: Hierarchical spatial planner (hierarchical_planner.py, jax_hierarchical_planner.py)
- Phase 4: F-aware planner (si_F_aware_lava.py) - coming soon
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

from .belief_utils import safe_belief_update, compute_vfe

# Emotional state tracking (Circumplex Model)
from .emotional_state import (
    EmotionalState,
    EmotionalStateTracker,
    compute_belief_entropy,
    compute_valence,
    compute_utility,
    compute_expected_utility,
    compute_empathic_emotional_state,
)

# Hierarchical planners
from .hierarchical_planner import (
    SpatialZone,
    ZonedLayout,
    ZoneAction,
    HierarchicalEmpathicPlanner,
    get_zoned_layout,
    has_zoned_layout,
    create_vertical_bottleneck_zones,
    create_symmetric_bottleneck_zones,
    create_narrow_zones,
)

from .jax_hierarchical_planner import (
    JaxZonedLayout,
    JaxHierarchicalPlanner,
    HierarchicalEmpathicPlannerJax,
    get_jax_zoned_layout,
    has_jax_zoned_layout,
    high_level_plan_jax,
    low_level_plan_jax,
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
    # Phase 3: Hierarchical (NumPy)
    "SpatialZone",
    "ZonedLayout",
    "ZoneAction",
    "HierarchicalEmpathicPlanner",
    "get_zoned_layout",
    "has_zoned_layout",
    "create_vertical_bottleneck_zones",
    "create_symmetric_bottleneck_zones",
    "create_narrow_zones",
    # Phase 3: Hierarchical (JAX)
    "JaxZonedLayout",
    "JaxHierarchicalPlanner",
    "HierarchicalEmpathicPlannerJax",
    "get_jax_zoned_layout",
    "has_jax_zoned_layout",
    "high_level_plan_jax",
    "low_level_plan_jax",
    # Utilities
    "safe_belief_update",
    "compute_vfe",
    # Emotional state (Circumplex Model)
    "EmotionalState",
    "EmotionalStateTracker",
    "compute_belief_entropy",
    "compute_valence",
    "compute_utility",
    "compute_expected_utility",
    "compute_empathic_emotional_state",
]
