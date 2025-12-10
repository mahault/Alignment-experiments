"""
Alignment Experiments - Active Inference with Theory of Mind and Path Flexibility.

This package provides tools for studying coordination and alignment in multi-agent systems.
"""

# Configuration
from .config import (
    PerformanceConfig,
    get_performance_config,
    set_performance_config,
    use_jax,
    enable_jax,
    disable_jax,
)

__version__ = "0.2.0"

__all__ = [
    "PerformanceConfig",
    "get_performance_config",
    "set_performance_config",
    "use_jax",
    "enable_jax",
    "disable_jax",
]
