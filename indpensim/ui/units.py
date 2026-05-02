"""UI-side temperature unit conversion.

The Recipe dataclass and JSON format store ``T_sp`` in Kelvin to match
the simulator core (MATLAB convention, used by the ODE). Engineers
author and read temperatures in Celsius. This module is the single
boundary that converts between the two — every UI widget that shows or
accepts a temperature value routes through here.
"""
from __future__ import annotations

_ABS_ZERO_C_IN_K = 273.15


def k_to_c(kelvin: float) -> float:
    return kelvin - _ABS_ZERO_C_IN_K


def c_to_k(celsius: float) -> float:
    return celsius + _ABS_ZERO_C_IN_K


def format_temp_c(kelvin: float, decimals: int = 1) -> str:
    """Render a stored-Kelvin value as a Celsius display string."""
    return f"{k_to_c(kelvin):.{decimals}f}°C"
