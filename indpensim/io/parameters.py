"""Model parameters for IndPenSim — grouped Python port of Parameter_list.m.

Original is a flat 105-element vector indexed as par(1..105). This module
exposes the same numbers as nine grouped frozen dataclasses for readability,
plus a `.to_legacy_par_vector()` method that returns the legacy flat array
in the original order — used by the line-by-line ODE port (`ode/rhs.py`)
so it can mirror MATLAB's `par(20)` etc. without renaming arithmetic.

Naming cleanups vs the original (per docs/parameters.md):
- Y_sx          → Y_sX             (consistent with ODE usage)
- delta_c_o     → delta_c_0        (the source declares 'o', ODE reads '0')
- k_3           → k3               (ODE uses k3)
- dealta_H_evap → delta_H_evap     (typo)
- pho_*         → rho_*            (typo for Greek rho)

Five parameters are mutated at runtime inside the ODE (mu_p, mux_max, c_s,
c_oil, abc, PAA_c, Tcin, O_2_in via disturbance flag; mu_h via inhibition
flag; K_diff via age modulation). Those mutations live in `ode/rhs.py`,
not here. The dataclasses are frozen so accidental mutation raises.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np


# ---------------------------------------------------------------------------
# Section 1 — Penicillin & biomass kinetics                       par(1..22)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KineticParameters:
    mu_p: float                          # par 1  — from x0
    mux_max: float                       # par 2  — from x0
    ratio_mu_e_mu_b: float = 0.4         # par 3
    P_std_dev: float = 0.0015            # par 4
    mean_P: float = 0.002                # par 5
    mu_v: float = 1.71e-4                # par 6
    mu_a: float = 3.5e-3                 # par 7
    mu_diff: float = 5.36e-3             # par 8
    beta_1: float = 0.006                # par 9
    K_b: float = 0.05                    # par 10
    K_diff: float = 0.75                 # par 11  (age-modulated at runtime)
    K_diff_L: float = 0.09               # par 12
    K_e: float = 0.009                   # par 13
    K_v: float = 0.05                    # par 14
    delta_r: float = 0.75e-4             # par 15
    k_v: float = 3.22e-5                 # par 16
    D: float = 2.66e-11                  # par 17
    rho_a0: float = 0.35                 # par 18
    rho_d: float = 0.18                  # par 19
    mu_h: float = 0.003                  # par 20  (overwritten under inhib_flag != 0)
    r_0: float = 1.5e-4                  # par 21
    delta_0: float = 1e-4                # par 22


# ---------------------------------------------------------------------------
# Section 2 — Stoichiometry, gas/oxygen transfer                  par(23..46)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessParameters:
    alpha_kla: float                     # par 31  — passed in, no default
    Y_sX: float = 1.85                   # par 23
    Y_sP: float = 0.9                    # par 24
    m_s: float = 0.029                   # par 25
    c_oil: float = 1000.0                # par 26  (+disturbance)
    c_s: float = 600.0                   # par 27  (+disturbance)
    Y_O2_X: float = 650.0                # par 28
    Y_O2_P: float = 160.0                # par 29
    m_O2_X: float = 17.5                 # par 30
    # par 31 is alpha_kla (positional, above)
    kla_a: float = 0.38                  # par 32  (renamed from `a`)
    kla_b: float = 0.34                  # par 33  (renamed from `b`)
    kla_c: float = -0.38                 # par 34  (renamed from `c`)
    kla_d: float = 0.25                  # par 35  (renamed from `d`)
    Henrys_c: float = 0.0251             # par 36
    n_imp: float = 3.0                   # par 37
    r: float = 2.1                       # par 38  (vessel radius, m)
    r_imp: float = 0.85                  # par 39
    Po: float = 5.0                      # par 40
    epsilon: float = 0.1                 # par 41
    g: float = 9.81                      # par 42
    R: float = 8.314                     # par 43
    X_crit_DO2: float = 0.1              # par 44
    P_crit_DO2: float = 0.3              # par 45
    A_inhib: float = 1.0                 # par 46


# ---------------------------------------------------------------------------
# Section 3 — Thermal                                       par(47..61, 94, 95, 103..105)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThermalParameters:
    Tf: float = 288.0                    # par 47
    Tw: float = 288.0                    # par 48
    Tcin: float = 285.0                  # par 49  (+disturbance)
    Th: float = 333.0                    # par 50
    Tair: float = 290.0                  # par 51
    C_ps: float = 5.9                    # par 52  kJ/kg/K
    C_pw: float = 4.18                   # par 53  kJ/g/K (units suspicious but preserved)
    delta_H_evap: float = 2430.7         # par 54  (typo "dealta" → "delta")
    U_jacket: float = 36.0               # par 55
    A_c: float = 105.0                   # par 56
    Eg: float = 1.488e4                  # par 57
    Ed: float = 1.7325e5                 # par 58
    k_g: float = 450.0                   # par 59
    k_d: float = 0.25e30                 # par 60
    Y_QX: float = 25.0                   # par 61
    alpha_evp: float = 5.24e-4           # par 94
    beta_T: float = 2.88                 # par 95
    Tv: float = 373.0                    # par 103
    T0: float = 273.0                    # par 104
    alpha_1: float = 2451.8              # par 105


# ---------------------------------------------------------------------------
# Section 4 — pH & ions                                          par(62..67)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PHIonParameters:
    abc: float = 0.033                   # par 62  (+disturbance)
    gamma1: float = 0.0325e-5            # par 63
    gamma2: float = 2.5e-11              # par 64
    m_ph: float = 0.0025                 # par 65
    K1: float = 1e-5                     # par 66
    K2: float = 2.5e-8                   # par 67


# ---------------------------------------------------------------------------
# Section 5 — Nitrogen budget                                    par(68..74)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class NitrogenParameters:
    N_conc_paa: float                    # par 69  — passed in
    N_conc_oil: float = 20000.0          # par 68
    N_conc_shot: float = 400000.0        # par 70
    Y_NX: float = 10.0                   # par 71
    Y_NP: float = 80.0                   # par 72
    m_N: float = 0.03                    # par 73
    X_crit_N: float = 150.0              # par 74


# ---------------------------------------------------------------------------
# Section 6 — PAA                                                par(75..80)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PAAParameters:
    PAA_c: float                         # par 75  — passed in
    Y_PAA_P: float = 187.5               # par 76
    Y_PAA_X: float = 37.5 * 1.2          # par 77  (= 45.0)
    m_PAA: float = 1.05                  # par 78
    X_crit_PAA: float = 2400.0           # par 79
    P_crit_PAA: float = 200.0            # par 80


# ---------------------------------------------------------------------------
# Section 7 — k4 polynomial (T·pH on hydrolysis)                 par(81..85)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class K4PolynomialParameters:
    B_1: float = -64.29                  # par 81
    B_2: float = -1.825                  # par 82
    B_3: float = 0.3649                  # par 83
    B_4: float = 0.1280                  # par 84
    B_5: float = -4.9496e-4              # par 85


# ---------------------------------------------------------------------------
# Section 8 — CO₂ & viscosity                                    par(86..93)
# (alpha_evp/beta_T sit in ThermalParameters as par 94/95)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CO2ViscosityParameters:
    delta_c_0: float = 0.89              # par 86  (was delta_c_o in source)
    k3: float = 0.005                    # par 87  (was k_3 in source)
    k1: float = 0.001                    # par 88
    k2: float = 0.0001                   # par 89
    t1: float = 1.0                      # par 90
    t2: float = 250.0                    # par 91
    q_co2: float = 0.123 * 1.1           # par 92  (= 0.1353)
    X_crit_CO2: float = 7570.0           # par 93


# ---------------------------------------------------------------------------
# Section 9 — Densities & inlet gas                             par(96..102)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DensityParameters:
    rho_g: float = 1540.0                # par 96  (was pho_g)
    rho_oil: float = 900.0               # par 97
    rho_w: float = 1000.0                # par 98
    rho_paa: float = 1000.0              # par 99
    O_2_in: float = 0.21                 # par 100  (+disturbance)
    N2_in: float = 0.79                  # par 101
    C_CO2_in: float = 0.033              # par 102


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Parameters:
    """Full 105-parameter set, grouped semantically.

    Use `Parameters.default(mu_p=..., mux_max=..., ...)` to build with the
    five runtime-supplied values and all other defaults from Parameter_list.m.
    Use `to_legacy_par_vector()` to flatten back to the original par(1..105)
    array consumed by the line-by-line ODE port.
    """

    kinetics: KineticParameters
    process: ProcessParameters
    thermal: ThermalParameters
    ph_ion: PHIonParameters
    nitrogen: NitrogenParameters
    paa: PAAParameters
    k4: K4PolynomialParameters
    co2_visc: CO2ViscosityParameters
    density: DensityParameters

    N_PARAMS: ClassVar[int] = 105

    @classmethod
    def default(
        cls,
        mu_p: float,
        mux_max: float,
        alpha_kla: float,
        N_conc_paa: float,
        PAA_c: float,
    ) -> "Parameters":
        """Mirrors Parameter_list(x0, alpha_kla, N_conc_paa, PAA_c) in the source."""
        return cls(
            kinetics=KineticParameters(mu_p=mu_p, mux_max=mux_max),
            process=ProcessParameters(alpha_kla=alpha_kla),
            thermal=ThermalParameters(),
            ph_ion=PHIonParameters(),
            nitrogen=NitrogenParameters(N_conc_paa=N_conc_paa),
            paa=PAAParameters(PAA_c=PAA_c),
            k4=K4PolynomialParameters(),
            co2_visc=CO2ViscosityParameters(),
            density=DensityParameters(),
        )

    def to_legacy_par_vector(self) -> np.ndarray:
        """Return the 105-element float array in original par(1..105) order.

        Use this from `ode/rhs.py` so a line-by-line port can write
        ``par[19]`` and reference the same value as MATLAB ``par(20)``.
        """
        k = self.kinetics
        p = self.process
        t = self.thermal
        ph = self.ph_ion
        n = self.nitrogen
        a = self.paa
        b4 = self.k4
        cv = self.co2_visc
        d = self.density
        vec = np.array([
            # 1..22
            k.mu_p, k.mux_max, k.ratio_mu_e_mu_b, k.P_std_dev, k.mean_P,
            k.mu_v, k.mu_a, k.mu_diff, k.beta_1, k.K_b,
            k.K_diff, k.K_diff_L, k.K_e, k.K_v, k.delta_r,
            k.k_v, k.D, k.rho_a0, k.rho_d, k.mu_h,
            k.r_0, k.delta_0,
            # 23..46
            p.Y_sX, p.Y_sP, p.m_s, p.c_oil, p.c_s,
            p.Y_O2_X, p.Y_O2_P, p.m_O2_X, p.alpha_kla, p.kla_a,
            p.kla_b, p.kla_c, p.kla_d, p.Henrys_c, p.n_imp,
            p.r, p.r_imp, p.Po, p.epsilon, p.g,
            p.R, p.X_crit_DO2, p.P_crit_DO2, p.A_inhib,
            # 47..61
            t.Tf, t.Tw, t.Tcin, t.Th, t.Tair,
            t.C_ps, t.C_pw, t.delta_H_evap, t.U_jacket, t.A_c,
            t.Eg, t.Ed, t.k_g, t.k_d, t.Y_QX,
            # 62..67
            ph.abc, ph.gamma1, ph.gamma2, ph.m_ph, ph.K1, ph.K2,
            # 68..74
            n.N_conc_oil, n.N_conc_paa, n.N_conc_shot,
            n.Y_NX, n.Y_NP, n.m_N, n.X_crit_N,
            # 75..80
            a.PAA_c, a.Y_PAA_P, a.Y_PAA_X, a.m_PAA, a.X_crit_PAA, a.P_crit_PAA,
            # 81..85
            b4.B_1, b4.B_2, b4.B_3, b4.B_4, b4.B_5,
            # 86..93
            cv.delta_c_0, cv.k3, cv.k1, cv.k2, cv.t1, cv.t2, cv.q_co2, cv.X_crit_CO2,
            # 94, 95
            t.alpha_evp, t.beta_T,
            # 96..102
            d.rho_g, d.rho_oil, d.rho_w, d.rho_paa, d.O_2_in, d.N2_in, d.C_CO2_in,
            # 103..105
            t.Tv, t.T0, t.alpha_1,
        ], dtype=float)
        if vec.size != self.N_PARAMS:
            raise RuntimeError(
                f"to_legacy_par_vector produced {vec.size} elements, expected {self.N_PARAMS}"
            )
        return vec
