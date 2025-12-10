"""
Global configuration for Alignment Experiments.

This module provides configuration settings that control runtime behavior,
especially performance optimizations like JAX acceleration.
"""

import os
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""

    # JAX acceleration toggle
    use_jax: bool = True  # Set to False to use NumPy (for debugging)

    # JIT compilation
    enable_jit_warmup: bool = True  # Warm up JAX on first use

    # GPU settings
    force_cpu: bool = False  # Set to True to disable GPU even if available

    # Memory management
    jax_memory_fraction: float = 0.75  # Fraction of GPU memory to use (0.0-1.0)

    # Logging
    verbose_jax: bool = False  # Log JAX compilation and execution details


# Global singleton instance
_performance_config = None


def get_performance_config() -> PerformanceConfig:
    """
    Get the global performance configuration.

    Returns
    -------
    PerformanceConfig
        Singleton performance configuration instance
    """
    global _performance_config

    if _performance_config is None:
        _performance_config = PerformanceConfig()

        # Apply environment variable overrides
        _apply_env_overrides(_performance_config)

        # Apply JAX platform settings
        _configure_jax_platform(_performance_config)

        # Log configuration
        LOGGER.info(f"Performance config initialized: use_jax={_performance_config.use_jax}")

    return _performance_config


def set_performance_config(config: PerformanceConfig):
    """
    Set the global performance configuration.

    Parameters
    ----------
    config : PerformanceConfig
        New configuration to use
    """
    global _performance_config
    _performance_config = config
    _configure_jax_platform(config)
    LOGGER.info(f"Performance config updated: use_jax={config.use_jax}")


def _apply_env_overrides(config: PerformanceConfig):
    """Apply environment variable overrides to config."""
    # JAX toggle
    if "USE_JAX" in os.environ:
        use_jax = os.environ["USE_JAX"].lower() in ("1", "true", "yes")
        config.use_jax = use_jax
        LOGGER.info(f"USE_JAX environment variable detected: {use_jax}")

    # Force CPU
    if "JAX_FORCE_CPU" in os.environ:
        force_cpu = os.environ["JAX_FORCE_CPU"].lower() in ("1", "true", "yes")
        config.force_cpu = force_cpu
        LOGGER.info(f"JAX_FORCE_CPU environment variable detected: {force_cpu}")

    # Memory fraction
    if "JAX_MEMORY_FRACTION" in os.environ:
        try:
            mem_frac = float(os.environ["JAX_MEMORY_FRACTION"])
            config.jax_memory_fraction = mem_frac
            LOGGER.info(f"JAX_MEMORY_FRACTION environment variable detected: {mem_frac}")
        except ValueError:
            LOGGER.warning(f"Invalid JAX_MEMORY_FRACTION value: {os.environ['JAX_MEMORY_FRACTION']}")


def _configure_jax_platform(config: PerformanceConfig):
    """Configure JAX platform settings based on config."""
    if not config.use_jax:
        return

    # Force CPU if requested
    if config.force_cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
        LOGGER.info("JAX configured to use CPU only")

    # Set memory fraction
    if config.jax_memory_fraction < 1.0:
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(config.jax_memory_fraction)
        LOGGER.info(f"JAX memory fraction set to {config.jax_memory_fraction}")

    # Disable JAX verbose logging unless explicitly enabled
    if not config.verbose_jax:
        os.environ.setdefault("JAX_LOG_COMPILES", "0")


# Convenience function to check if JAX should be used
def use_jax() -> bool:
    """
    Check if JAX acceleration should be used.

    Returns
    -------
    bool
        True if JAX should be used, False otherwise
    """
    return get_performance_config().use_jax


# Convenience function to enable/disable JAX
def enable_jax():
    """Enable JAX acceleration globally."""
    config = get_performance_config()
    config.use_jax = True
    LOGGER.info("JAX acceleration enabled")


def disable_jax():
    """Disable JAX acceleration globally (fallback to NumPy)."""
    config = get_performance_config()
    config.use_jax = False
    LOGGER.info("JAX acceleration disabled (using NumPy)")


# Usage examples in docstring
"""
Usage Examples
==============

1. Default behavior (JAX enabled):
    from src.config import use_jax

    if use_jax():
        from src.metrics.jax_path_flexibility import compute_F_arrays_for_policies_jax as compute_F
    else:
        from src.metrics.path_flexibility import compute_F_arrays_for_policies as compute_F

2. Programmatically disable JAX:
    from src.config import disable_jax

    disable_jax()  # All subsequent calls will use NumPy

3. Environment variable control:
    # Disable JAX via environment variable
    export USE_JAX=0
    python my_script.py

    # Force CPU (no GPU)
    export JAX_FORCE_CPU=1
    python my_script.py

    # Limit GPU memory to 50%
    export JAX_MEMORY_FRACTION=0.5
    python my_script.py

4. Custom configuration:
    from src.config import set_performance_config, PerformanceConfig

    custom_config = PerformanceConfig(
        use_jax=True,
        force_cpu=True,
        jax_memory_fraction=0.5,
        verbose_jax=True
    )
    set_performance_config(custom_config)
"""
