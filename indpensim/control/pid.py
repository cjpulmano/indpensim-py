"""Incremental-form PID controller — port of PIDSimple3.m.

The derivative term acts on the measured value (PV), not the error, to
avoid derivative kicks on setpoint changes. Integral and derivative paths
short-circuit for very small time constants (Ti < 1e-7, Td <= 0.001) to
avoid division-by-zero and noise amplification on essentially-PI or
essentially-P controllers.

Update law (matches PIDSimple3.m line-by-line):

    P = e_k - e_{k-1}
    I = (e_k * h) / Ti           if Ti > 1e-7  else 0
    D = -(Td/h) * (y_k - 2*y_{k-1} + y_{k-2})   if Td > 0.001 else 0
    u = clip(u_{k-1} + Kp*(P + I + D), u_min, u_max)
"""
from __future__ import annotations


_TI_DEADBAND = 1e-7
_TD_DEADBAND = 0.001


def pid_step(
    *,
    uk1: float,
    ek: float,
    ek1: float,
    yk: float,
    yk1: float,
    yk2: float,
    u_min: float,
    u_max: float,
    Kp: float,
    Ti: float,
    Td: float,
    h: float,
) -> float:
    """Compute one step of the incremental PID law and return the saturated control u_k.

    All arguments are keyword-only to mirror the explicit naming in the MATLAB
    function signature and prevent positional mistakes (12 args is too many to
    pass by position safely).
    """
    P = ek - ek1
    I = (ek * h) / Ti if Ti > _TI_DEADBAND else 0.0
    D = -(Td / h) * (yk - 2.0 * yk1 + yk2) if Td > _TD_DEADBAND else 0.0
    u = uk1 + Kp * (P + I + D)
    if u > u_max:
        u = u_max
    if u < u_min:
        u = u_min
    return u
