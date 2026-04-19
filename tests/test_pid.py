"""Tests for the PID controller — exercising every branch of PIDSimple3.m."""
from __future__ import annotations

import pytest

from indpensim.control.pid import pid_step


def test_pure_proportional_with_no_integral_no_derivative():
    """Ti < 1e-7 and Td <= 0.001 → only the P term contributes."""
    u = pid_step(
        uk1=10.0, ek=2.0, ek1=1.5, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=3.0, Ti=0.0, Td=0.0, h=0.1,
    )
    # P = ek - ek1 = 0.5; u = 10 + 3*0.5 = 11.5
    assert u == pytest.approx(11.5, rel=1e-12)


def test_proportional_plus_integral():
    u = pid_step(
        uk1=0.0, ek=2.0, ek1=2.0, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=4.0, Ti=2.0, Td=0.0, h=0.5,
    )
    # P = 0; I = 2*0.5/2 = 0.5; u = 0 + 4*(0+0.5+0) = 2.0
    assert u == pytest.approx(2.0, rel=1e-12)


def test_proportional_plus_derivative_on_pv():
    """Derivative is computed on yk, not on the error."""
    u = pid_step(
        uk1=5.0, ek=0.0, ek1=0.0, yk=10.0, yk1=8.0, yk2=7.0,
        u_min=-1e9, u_max=1e9, Kp=2.0, Ti=0.0, Td=1.0, h=0.5,
    )
    # P=0; D = -(1.0/0.5) * (10 - 2*8 + 7) = -2 * 1 = -2; u = 5 + 2*(-2) = 1.0
    assert u == pytest.approx(1.0, rel=1e-12)


def test_full_pid_combination_matches_hand_derivation():
    u = pid_step(
        uk1=100.0, ek=3.0, ek1=2.5, yk=20.0, yk1=18.0, yk2=15.0,
        u_min=0.0, u_max=200.0, Kp=5.0, Ti=10.0, Td=2.0, h=1.0,
    )
    # P = 0.5
    # I = 3*1/10 = 0.3
    # D = -(2/1) * (20 - 36 + 15) = -2 * -1 = 2
    # u = 100 + 5*(0.5 + 0.3 + 2) = 100 + 14 = 114
    assert u == pytest.approx(114.0, rel=1e-12)


def test_upper_saturation_clamps_to_u_max():
    u = pid_step(
        uk1=180.0, ek=10.0, ek1=0.0, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=0.0, u_max=200.0, Kp=10.0, Ti=0.0, Td=0.0, h=1.0,
    )
    # uk1 + Kp*P = 180 + 10*10 = 280 → clamped to 200
    assert u == 200.0


def test_lower_saturation_clamps_to_u_min():
    u = pid_step(
        uk1=10.0, ek=-5.0, ek1=5.0, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=0.0, u_max=200.0, Kp=2.0, Ti=0.0, Td=0.0, h=1.0,
    )
    # uk1 + Kp*P = 10 + 2*(-10) = -10 → clamped to 0
    assert u == 0.0


def test_integral_deadband_just_above_threshold_activates():
    """Ti just above the 1e-7 threshold should engage the integral term."""
    u_off = pid_step(
        uk1=0.0, ek=1.0, ek1=1.0, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=1.0, Ti=1e-8, Td=0.0, h=1.0,
    )
    u_on = pid_step(
        uk1=0.0, ek=1.0, ek1=1.0, yk=0.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=1.0, Ti=2e-7, Td=0.0, h=1.0,
    )
    assert u_off == 0.0          # Ti < threshold → no I
    assert u_on != 0.0           # Ti > threshold → I term active


def test_derivative_deadband_just_above_threshold_activates():
    u_off = pid_step(
        uk1=0.0, ek=0.0, ek1=0.0, yk=10.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=1.0, Ti=0.0, Td=0.0005, h=1.0,
    )
    u_on = pid_step(
        uk1=0.0, ek=0.0, ek1=0.0, yk=10.0, yk1=0.0, yk2=0.0,
        u_min=-1e9, u_max=1e9, Kp=1.0, Ti=0.0, Td=0.002, h=1.0,
    )
    assert u_off == 0.0          # Td <= threshold → no D
    assert u_on != 0.0           # Td > threshold → D term active
