"""Port of fctrl_indpensim.m — manipulated-variable controller.

Single-function line-by-line port (mirrors MATLAB structure). Will refactor
into pH/temperature/SBC/faults/PAA-PID submodules as a Phase 4 cleanup once
the end-to-end batch passes.

Indexing: 1-based ``k`` (MATLAB-native). At call time, the state-channel
arrays in ``history`` must be populated for samples ``1..k-1`` and (for
the very first MATLAB sample) slot 1 is pre-filled with ``x0``-derived
values per ``indpensim.m:113-122``. Controller-output channels (Fa, Fb,
Fc, Fh, Fs, Fpaa, ...) must be populated for samples ``1..k-1`` so PID
loops can see ``u_{k-1}``.

Faithful-port quirks (verified against MATLAB source, *not* fixed):
  * Line 45 in fctrl: ``ph_err1`` is computed without the ``pH_sp -`` term.
    This is a published-source bug; we reproduce it verbatim. If trajectory
    diverges, this is the first place to look.
  * Line 147: heating-branch PID for Fh uses ``X.Fc.y(k-1)`` as ``u_prev``,
    not ``X.Fh.y(k-1)``. Ignores the prior heating output.
  * Lines 148-149: cooling carry-over in heating branch — ``Fc=0`` then
    immediately overwritten by ``Fc = X.Fc.y(k-1)*0.3``.
  * Lines 153-158: 1e-4 floor on Fc/Fh applied AFTER PID + carry-over.
  * Lines 207-209 and 305-307: PRBS branch overwrites Fs/Fpaa with the
    previous sample's value for k > 475 (held-flat behavior on the run-up
    to the noise window). Faithful even though weird.
  * ``rng('shuffle')``: MATLAB seeds from the wall clock here. Cannot be
    reproduced bit-for-bit. We seed numpy from time.time_ns(); validation
    against MATLAB requires PRBS=0 OR a captured noise trajectory.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from indpensim.control.history import BatchHistory, IndustrialData
from indpensim.control.pid import pid_step
from indpensim.io.initial_conditions import ControlFlags


# Sequential Batch Control recipes — verbatim from fctrl_indpensim.m:178-289.
_RECIPE_FS_K       = (15, 60, 80, 100, 120, 140, 160, 180, 200, 220,
                      240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750)
_RECIPE_FS_SP      = ( 8, 15, 30,  75, 150,  30,  37,  43,  47,  51,
                       57,  61,  65,  72,  76,  80,  84,  90, 116,  90,   80)

_RECIPE_FOIL_K     = (20, 80, 280, 300, 320, 340, 360, 380, 400, 1750)
_RECIPE_FOIL_SP    = (22, 30,  35,  34,  33,  32,  31,  30,  29,   23)

_RECIPE_FG_K       = (40, 100, 200, 450, 1000, 1250, 1750)
_RECIPE_FG_SP      = (30,  42,  55,  60,   75,   65,   60)

_RECIPE_PRES_K     = (62.5, 125, 150, 200, 500, 750, 1000, 1750)
_RECIPE_PRES_SP    = (0.6, 0.7, 0.8, 0.9, 1.1, 1.0,  0.9,  0.9)

_RECIPE_DISCH_K    = (500, 510, 650, 660, 750, 760, 850, 860, 950, 960,
                      1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750)
_RECIPE_DISCH_SP   = (   0, 4000,    0, 4000,    0, 4000,   0, 4000,   0, 4000,
                         0, 4000,    0, 4000,    0, 4000,   0, 4000,    0)

_RECIPE_WATER_K    = (250, 375, 750, 800, 850, 1000, 1250, 1350, 1750)
_RECIPE_WATER_SP   = (  0, 500, 100,   0, 400,  150,  250,    0,  100)

_RECIPE_PAA_K      = (25, 200, 1000, 1500, 1750)
_RECIPE_PAA_SP     = ( 5,   0,   10,    4,    0)


def _recipe_lookup(k: int, breakpoints: tuple, setpoints: tuple) -> float:
    """Reproduces fctrl_indpensim.m's loop:
        for SQ=1:length(R), if k<=R(SQ), Fs=Sp(SQ); break; else Fs=Sp(end);
    Returns the FIRST setpoint whose breakpoint k <= R(SQ) holds; otherwise
    the last setpoint.
    """
    for bp, sp in zip(breakpoints, setpoints):
        if k <= bp:
            return float(sp)
    return float(setpoints[-1])


def _ramp_fault_value(k: int, fault_value: float) -> float:
    """Mirror of the per-sample interp in pH (29-34) and temp (94-100) sensor faults.

    MATLAB:
        Ramp_function = [ 0 0; 200 0; 800 V; 1750 V];
        tInterp = 1:1:1750;
        Ramp_function_interp = interp1(...,'linear','extrap');
        sensor_error = Ramp_function_interp(k);
    """
    xs = np.array([0.0, 200.0, 800.0, 1750.0])
    ys = np.array([0.0, 0.0, fault_value, fault_value])
    return float(np.interp(k, xs, ys))


@dataclass
class ControllerOutputs:
    """Manipulated-variable vector — MATLAB ``u`` struct."""
    Fg: float
    RPM: float
    Fs: float
    Fa: float
    Fb: float
    Fc: float
    Fh: float
    d1: int                # acid/base on/off
    tfl: int               # heat/cool on/off
    Fw: float
    pressure: float
    viscosity: float
    Fremoved: float
    Fpaa: float
    Foil: float
    NH3_shots: float
    Fault_ref: int


def _ph_from_h_plus(h_plus: float) -> float:
    """``-log10([H+])``, matching MATLAB ``-log(X.pH.y(k))/log(10)``."""
    return -math.log(h_plus) / math.log(10.0)


def controller_step(
    history: BatchHistory,
    Xd: IndustrialData,
    k: int,
    h: float,
    T: float,
    ctrl_flags: ControlFlags,
    *,
    rng: np.random.Generator | None = None,
) -> ControllerOutputs:
    """Compute manipulated variables for one MATLAB sample ``k`` (1-based).

    Mutates ``history.channels['PRBS_noise_addition'][k]`` (the only
    documented controller side-effect). Reads state channels at indices
    ``[k-3 .. k-1]`` and control-output channels at ``[k-1]``. Returns the
    ``u`` struct equivalent.

    ``rng`` is used only when ``ctrl_flags.PRBS == 1``; pass an explicit
    ``np.random.default_rng()`` for testability. When ``None`` and PRBS
    fires, we fall back to a wall-clock seeded RNG (faithful to MATLAB
    ``rng('shuffle')`` behavior, but not reproducible).
    """
    Faults = ctrl_flags.Faults

    # ============================== pH controller ============================
    pH_sensor_error = 0.0
    if Faults == 8:
        pH_sensor_error = _ramp_fault_value(k, 0.1)

    pH_sp = ctrl_flags.pH_sp
    # Error history (fctrl lines 40-46). Note line 45 is a *typo* in MATLAB
    # — ph_err1 is missing the (pH_sp -) term. Faithful port:
    if k == 1 or k == 2:
        ph_err  = pH_sp - _ph_from_h_plus(history.y("pH", 1)) + pH_sensor_error
        ph_err1 = pH_sp - _ph_from_h_plus(history.y("pH", 1)) + pH_sensor_error
    else:
        ph_err  = pH_sp - _ph_from_h_plus(history.y("pH", k - 1)) + pH_sensor_error
        # MATLAB line 45 — no pH_sp term, intentionally faithful
        ph_err1 = - _ph_from_h_plus(history.y("pH", k - 2)) + pH_sensor_error

    # pH PV history (lines 49-61)
    if k == 1 or k == 2:
        ph  = _ph_from_h_plus(history.y("pH", 1))
        ph1 = _ph_from_h_plus(history.y("pH", 1))
        ph2 = _ph_from_h_plus(history.y("pH", 1))
    elif k == 3:
        ph  = _ph_from_h_plus(history.y("pH", 2))
        ph1 = _ph_from_h_plus(history.y("pH", 1))
        ph2 = _ph_from_h_plus(history.y("pH", 1))
    else:
        ph  = _ph_from_h_plus(history.y("pH", k - 1))
        ph1 = _ph_from_h_plus(history.y("pH", k - 2))
        ph2 = _ph_from_h_plus(history.y("pH", k - 3))

    # Branch on whether base or acid is needed (lines 64-90)
    if ph_err >= -0.05:
        ph_on_off = 1
        prev_Fb = history.y("Fb", 1) if k == 1 else history.y("Fb", k - 1)
        Fb = pid_step(
            uk1=prev_Fb, ek=ph_err, ek1=ph_err1,
            yk=ph, yk1=ph1, yk2=ph2,
            u_min=0.0, u_max=225.0,
            Kp=8e-2, Ti=4.0000e-05, Td=8, h=h,
        )
        Fa = 0.0
    elif ph_err <= -0.05:
        ph_on_off = 1
        if k == 1:
            Fa = pid_step(
                uk1=history.y("Fa", 1), ek=ph_err, ek1=ph_err1,
                yk=ph, yk1=ph1, yk2=ph2,
                u_min=0.0, u_max=225.0,
                Kp=8e-2, Ti=12.5, Td=0.125, h=h,
            )
            Fb = 0.0
        else:
            Fa = pid_step(
                uk1=history.y("Fa", k - 1), ek=ph_err, ek1=ph_err1,
                yk=ph, yk1=ph1, yk2=ph2,
                u_min=0.0, u_max=225.0,
                Kp=8e-2, Ti=12.5, Td=0.125, h=h,
            )
            # Carry-over of base flow as it gets washed out (line 83)
            Fb = history.y("Fb", k - 1) * 0.5
    else:
        ph_on_off = 0
        Fb = 0.0
        Fa = 0.0

    # ============================== Temperature controller ===================
    T_sensor_error = 0.0
    if Faults == 7:
        T_sensor_error = _ramp_fault_value(k, 0.4)

    T_sp = ctrl_flags.T_sp
    if k == 1 or k == 2:
        temp_err  = T_sp - history.y("T", 1) + T_sensor_error
        temp_err1 = T_sp - history.y("T", 1) + T_sensor_error
    else:
        temp_err  = T_sp - history.y("T", k - 1) + T_sensor_error
        temp_err1 = T_sp - history.y("T", k - 2) + T_sensor_error

    if k == 1 or k == 2:
        temp  = history.y("T", 1)
        temp1 = history.y("T", 1)
        temp2 = history.y("T", 1)
    elif k == 3:
        temp  = history.y("T", 2)
        temp1 = history.y("T", 1)
        temp2 = history.y("T", 1)
    else:
        temp  = history.y("T", k - 1)
        temp1 = history.y("T", k - 2)
        temp2 = history.y("T", k - 3)

    # Heat/cool branch (lines 130-151)
    if temp_err <= 0.05:
        temp_on_off = 0   # cooling
        prev_Fc = history.y("Fc", 1) if k == 1 else history.y("Fc", k - 1)
        Fc = pid_step(
            uk1=prev_Fc, ek=temp_err, ek1=temp_err1,
            yk=temp, yk1=temp1, yk2=temp2,
            u_min=0.0, u_max=1.5e3,
            Kp=-300, Ti=1.6, Td=0.005, h=h,
        )
        Fh = 0.0 if k == 1 else history.y("Fh", k - 1) * 0.1
    else:
        temp_on_off = 1   # heating
        # Note line 144,147: u_prev arg is X.Fc.y(...) — NOT X.Fh — verbatim.
        prev_for_Fh = history.y("Fc", 1) if k == 1 else history.y("Fc", k - 1)
        Fh = pid_step(
            uk1=prev_for_Fh, ek=temp_err, ek1=temp_err1,
            yk=temp, yk1=temp1, yk2=temp2,
            u_min=0.0, u_max=1.5e3,
            Kp=50, Ti=0.050, Td=1, h=h,
        )
        # Lines 145-150: Fc = 0; (carry-over in else) Fc = X.Fc.y(k-1)*0.3
        Fc = 0.0 if k == 1 else history.y("Fc", k - 1) * 0.3

    # Numerical-stability floor (lines 153-158)
    if Fc < 1e-4:
        Fc = 1e-4
    if Fh < 1e-4:
        Fh = 1e-4

    # ============================== Sequential Batch Control =================
    # Captured batch uses SBC=0; SBC=1 path is a thin pass-through of Xd.
    if ctrl_flags.SBC == 1:
        Foil       = Xd.NH3_shots(k)        # placeholder — IndustrialData stub
        F_discharge = 0.0
        pressure   = 0.0
        Fpaa       = 0.0
        Fw         = 0.0
        viscosity  = 0.0
        Fg         = 0.0
        Fs         = 0.0
    else:
        # SBC=0 — recipe-driven setpoints
        viscosity = 4.0
        Fs = _recipe_lookup(k, _RECIPE_FS_K, _RECIPE_FS_SP)

        # Add PRBS to Fs (lines 193-216)
        if ctrl_flags.PRBS == 1:
            if k > 500 and k % 100 == 0:
                _rng = rng if rng is not None else np.random.default_rng(time.time_ns())
                # MATLAB: random_number = randi(3,1,1) -> {1,2,3} uniform
                # noise_factor = 15
                random_number = int(_rng.integers(1, 4))
                noise_factor = 15.0
                if random_number == 1:
                    random_noise = 0.0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                history.set("PRBS_noise_addition", k, random_noise)
            if k > 475:
                Fs = history.y("Fs", k - 1)
            if k > 500 and k % 100 == 0:
                # MATLAB: Fs = X.Fs.y(k-1) + X.PRBS_noise_addition(end)
                # "(end)" = the last assigned slot — the slot we just wrote at k.
                Fs = history.y("Fs", k - 1) + history.y("PRBS_noise_addition", k)
        else:
            history.set("PRBS_noise_addition", k, 0.0)

        Foil        = _recipe_lookup(k, _RECIPE_FOIL_K, _RECIPE_FOIL_SP)
        Fg          = _recipe_lookup(k, _RECIPE_FG_K, _RECIPE_FG_SP)
        pressure    = _recipe_lookup(k, _RECIPE_PRES_K, _RECIPE_PRES_SP)
        F_discharge = -_recipe_lookup(k, _RECIPE_DISCH_K, _RECIPE_DISCH_SP)
        Fw          = _recipe_lookup(k, _RECIPE_WATER_K, _RECIPE_WATER_SP)
        Fpaa        = _recipe_lookup(k, _RECIPE_PAA_K, _RECIPE_PAA_SP)

        if ctrl_flags.PRBS == 1:
            if k > 500 and k % 100 == 0:
                _rng = rng if rng is not None else np.random.default_rng(time.time_ns())
                random_number = int(_rng.integers(1, 4))
                noise_factor = 1.0
                if random_number == 1:
                    random_noise = 0.0
                elif random_number == 2:
                    random_noise = noise_factor
                else:
                    random_noise = -noise_factor
                history.set("PRBS_noise_addition", k, random_noise)
            if k > 475:
                Fpaa = history.y("Fpaa", k - 1)
            if k > 500 and k % 100 == 0:
                Fpaa = history.y("Fpaa", k - 1) + history.y("PRBS_noise_addition", k)

    # ============================== Process faults ===========================
    # Aeration fault
    Fault_ref = 0
    if Faults in (1, 6):
        if 100 <= k <= 120:
            Fg = 20.0
            Fault_ref = 1
        if 500 <= k <= 550:
            Fg = 20.0
            Fault_ref = 1
    if Faults in (2, 6):
        if 500 <= k <= 520:
            pressure = 2.0
            Fault_ref = 1
        if 1000 <= k <= 1200:
            pressure = 2.0
            Fault_ref = 1
    if Faults in (3, 6):
        if 100 <= k <= 150:
            Fs = 2.0
            Fault_ref = 1
        if 380 <= k <= 460:
            Fs = 20.0
            Fault_ref = 1
        if 1000 <= k <= 1070:
            Fs = 20.0
            Fault_ref = 1
    if Faults in (4, 6):
        if 400 <= k <= 420:
            Fb = 5.0
            Fault_ref = 1
        if 700 <= k <= 800:
            Fb = 10.0
            Fault_ref = 1
    if Faults in (5, 6):
        if 350 <= k <= 450:
            Fc = 2.0
            Fault_ref = 1
        if 1200 <= k <= 1350:
            Fc = 10.0
            Fault_ref = 1

    # ============================== PAA Raman PID (Raman_spec == 2) ==========
    if ctrl_flags.Raman_spec == 2:
        PAA_sp = 1200.0
        if k == 1 or k == 2:
            PAA_err  = PAA_sp - history.y("PAA", 1)
            PAA_err1 = PAA_sp - history.y("PAA", 1)
        else:
            PAA_err  = PAA_sp - history.y("PAA", k - 1)
            PAA_err1 = PAA_sp - history.y("PAA", k - 2)

        if k * h < 10:
            pass     # keep Fpaa from the recipe
        else:
            if k == 1 or k == 2:
                temp_p  = history.y("PAA_pred", 1)
                temp1_p = history.y("PAA_pred", 1)
                temp2_p = history.y("PAA_pred", 1)
            elif k == 3:
                temp_p  = history.y("PAA_pred", 2)
                temp1_p = history.y("PAA_pred", 1)
                temp2_p = history.y("PAA_pred", 1)
            else:
                temp_p  = history.y("PAA_pred", k - 2)
                temp1_p = history.y("PAA_pred", k - 3)
                temp2_p = history.y("PAA_pred", k - 4)

            prev_Fpaa = history.y("Fpaa", 1) if k == 1 else history.y("Fpaa", k - 1)
            Fpaa = pid_step(
                uk1=prev_Fpaa, ek=PAA_err, ek1=PAA_err1,
                yk=temp_p, yk1=temp1_p, yk2=temp2_p,
                u_min=0.0, u_max=150.0,
                Kp=0.1, Ti=0.50, Td=0.0 * 0.002, h=h,
            )

    # ============================== Output struct ============================
    return ControllerOutputs(
        Fg=Fg,
        RPM=100.0,
        Fs=Fs,
        Fa=Fa,
        Fb=Fb,
        Fc=Fc,
        Fh=Fh,
        d1=ph_on_off,
        tfl=temp_on_off,
        Fw=Fw,
        pressure=pressure,
        viscosity=viscosity,
        Fremoved=F_discharge,
        Fpaa=Fpaa,
        Foil=Foil,
        NH3_shots=Xd.NH3_shots(k),
        Fault_ref=Fault_ref,
    )
