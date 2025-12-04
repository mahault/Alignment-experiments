"""
ToM (Theory of Mind) modules for multi-agent active inference.

This package contains:
- si_tom_F_prior: F-aware policy prior for Experiment 2
"""

from .si_tom_F_prior import ToMPolicyConfig, run_tom_step_with_F_prior

__all__ = [
    "ToMPolicyConfig",
    "run_tom_step_with_F_prior",
]
