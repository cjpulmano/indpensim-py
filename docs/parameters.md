# Parameter glossary (`par` vector — 105 entries)

Source: `Parameter_list.m`. The function returns a flat vector `par(1..105)` consumed by `indpensim_ode.m`. All values listed below are *defaults* — five entries are mutated at runtime.

## Inputs to `Parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)`

| Argument | Used in par() slot | Source | Notes |
|----------|-------------------|--------|-------|
| `x0.mup`           | par(1)  `mu_p`            | initial state struct | Penicillin specific growth rate. |
| `x0.mux`           | par(2)  `mux_max`         | initial state struct | Max biomass specific growth rate. |
| `alpha_kla`        | par(31) `alpha_kla`       | controller            | kla correlation prefactor; varies by control mode. |
| `N_conc_paa`       | par(69) `N_conc_paa`      | controller            | Nitrogen concentration of PAA feed. |
| `PAA_c`            | par(75) `PAA_c`           | controller            | PAA inlet concentration. |

The remaining 100 parameters are hard-coded constants in this file.

## Section 1 — Penicillin/biomass kinetics (par 1–22)

| par | Symbol            | Value            | Units              | Where used |
|-----|-------------------|------------------|--------------------|------------|
| 1   | `mu_p`            | from x0          | h⁻¹                | `r_p` penicillin production rate |
| 2   | `mux_max`         | from x0          | h⁻¹                | `mu_a0`, `mu_e` biomass kinetics |
| 3   | `ratio_mu_e_mu_b` | 0.4              | —                  | `mu_a0 = ratio_mu_e_mu_b * mux_max * ...` |
| 4   | `P_std_dev`       | 0.0015           | g/L                | Penicillin Gaussian inhibition curve |
| 5   | `mean_P`          | 0.002            | g/L                | Center of `P_inhib` Gaussian |
| 6   | `mu_v`            | 1.71e-4          | g/g/h              | Vacuole formation rate |
| 7   | `mu_a`            | 3.5e-3           | h⁻¹                | Biomass autolysis rate |
| 8   | `mu_diff`         | 5.36e-3          | h⁻¹                | A0 → A1 differentiation |
| 9   | `beta_1`          | 0.006            | —                  | Age-dependent K_diff modifier |
| 10  | `K_b`             | 0.05             | g/L                | Saturation const for r_b0 (branching) |
| 11  | `K_diff`          | 0.75             | g/L                | A0→A1 differentiation half-sat (modulated by age) |
| 12  | `K_diff_L`        | 0.09             | g/L                | Lower bound on K_diff after age modulation |
| 13  | `K_e`             | 0.009            | g/L                | Half-sat for extension r_e1 |
| 14  | `K_v`             | 0.05             | g/L                | Half-sat for vacuole formation |
| 15  | `delta_r`         | 7.5e-5           | cm                 | Vacuole-size grid spacing (FD) |
| 16  | `k_v`             | 3.22e-5          | cm/h               | Vacuole growth velocity (advection coef) |
| 17  | `D`               | 2.66e-11         | cm²/h              | Vacuole-size diffusion coef |
| 18  | `rho_a0`          | 0.35             | g/cm³              | Density of branching biomass |
| 19  | `rho_d`           | 0.18             | g/cm³              | Density of degenerated biomass |
| 20  | `mu_h`            | 0.003            | h⁻¹                | **Recomputed every step** under inhib_flag 1 or 2 — see gotcha #3. |
| 21  | `r_0`             | 1.5e-4           | cm                 | Initial vacuole radius |
| 22  | `delta_0`         | 1e-4             | cm                 | Vacuole birth-size offset |

## Section 2 — Stoichiometry & gas/oxygen transfer (par 23–46)

| par | Symbol         | Value     | Units            | Notes |
|-----|----------------|-----------|------------------|-------|
| 23  | `Y_sX`         | 1.85      | g/g              | Substrate yield on biomass (`Y_sx` in source — note casing) |
| 24  | `Y_sP`         | 0.9       | g/g              | Substrate yield on penicillin |
| 25  | `m_s`          | 0.029     | g/g/h            | Substrate maintenance |
| 26  | `c_oil`        | 1000      | g/L              | Oil feed concentration; +disturbance |
| 27  | `c_s`          | 600       | g/L              | Sugar feed concentration; +disturbance |
| 28  | `Y_O2_X`       | 650       | mg/g             | Oxygen yield on biomass |
| 29  | `Y_O2_P`       | 160       | mg/g             | Oxygen yield on penicillin |
| 30  | `m_O2_X`       | 17.5      | mg/g             | Oxygen maintenance |
| 31  | `alpha_kla`    | passed in | —                | kla correlation prefactor |
| 32  | `a`            | 0.38      | —                | kla exponent on superficial velocity V_s |
| 33  | `b`            | 0.34      | —                | kla exponent on Pt/V_m **(name collision — see gotcha #4)** |
| 34  | `c`            | -0.38     | —                | kla exponent on (viscosity/100) |
| 35  | `d`            | 0.25      | —                | kla exponent on oil fraction |
| 36  | `Henrys_c`     | 0.0251    | bar·L/mg         | Henry's constant for O₂ |
| 37  | `n_imp`        | 3         | —                | Number of impellers |
| 38  | `r`            | 2.1       | m                | Vessel radius |
| 39  | `r_imp`        | 0.85      | m                | Impeller radius |
| 40  | `Po`           | 5         | —                | Power number |
| 41  | `epsilon`      | 0.1       | —                | Gas holdup fraction |
| 42  | `g`            | 9.81      | m/s²             | Gravity (yes, hard-coded) |
| 43  | `R`            | 8.314     | J/mol/K          | Universal gas constant |
| 44  | `X_crit_DO2`   | 0.1       | %                | DO₂ critical for biomass inhibition |
| 45  | `P_crit_DO2`   | 0.3       | %                | DO₂ critical for penicillin inhibition |
| 46  | `A_inhib`      | 1         | —                | Slope of all `tanh()` inhibition functions |

## Section 3 — Thermal (par 47–61)

| par | Symbol           | Value      | Units            | Notes |
|-----|------------------|------------|------------------|-------|
| 47  | `Tf`             | 288        | K                | Feed temperature |
| 48  | `Tw`             | 288        | K                | Water-for-injection temperature |
| 49  | `Tcin`           | 285        | K                | Coolant inlet temp; +disturbance |
| 50  | `Th`             | 333        | K                | Heating water inlet temp |
| 51  | `Tair`           | 290        | K                | Ambient air |
| 52  | `C_ps`           | 5.9        | kJ/kg/K          | Specific heat of broth/feed |
| 53  | `C_pw`           | 4.18       | kJ/g/K           | Specific heat of water (units suspicious — likely meant kJ/kg/K, but consume as-is) |
| 54  | `dealta_H_evap`  | 2430.7     | kJ/kg            | Evaporation enthalpy (typo "dealta" preserved) |
| 55  | `U_jacket`       | 36         | kW/m²/K          | Jacket overall HTC |
| 56  | `A_c`            | 105        | m²               | Jacket area |
| 57  | `Eg`             | 1.488e4    | J/mol            | Activation E (growth) — used in T_inhib |
| 58  | `Ed`             | 1.7325e5   | J/mol            | Activation E (death) |
| 59  | `k_g`            | 450        | —                | Pre-exp factor (growth) |
| 60  | `k_d`            | 0.25e30    | —                | Pre-exp factor (death) — yes, that's 2.5e29 |
| 61  | `Y_QX`           | 25         | kJ/g             | Heat of reaction per biomass |

## Section 4 — pH & ions (par 62–67)

| par | Symbol      | Value      | Units            | Notes |
|-----|-------------|------------|------------------|-------|
| 62  | `abc`       | 0.033      | mol/L            | Acid/base feed concentration; +disturbance |
| 63  | `gamma1`    | 3.25e-7    | mol H⁺/g         | Stoichiometric H⁺/OH⁻ from biomass kinetics |
| 64  | `gamma2`    | 2.5e-11    | mol H⁺/g         | H⁺ from inflow (Fs/Fb/Fa/etc.) |
| 65  | `m_ph`      | 0.0025     | mol H⁺/g/h       | pH maintenance from total biomass |
| 66  | `K1`        | 1e-5       | mol/L            | pH inhibition lower break (~pKa1) |
| 67  | `K2`        | 2.5e-8     | mol/L            | pH inhibition upper break (~pKa2) |

## Section 5 — Nitrogen budget (par 68–74)

| par | Symbol         | Value    | Units             |
|-----|----------------|----------|-------------------|
| 68  | `N_conc_oil`   | 20000    | mg/L              |
| 69  | `N_conc_paa`   | passed in | mg/L              |
| 70  | `N_conc_shot`  | 400000   | mg/kg             |
| 71  | `Y_NX`         | 10       | mg N₂/g X         |
| 72  | `Y_NP`         | 80       | mg N₂/g P         |
| 73  | `m_N`          | 0.03     | g/L/h             |
| 74  | `X_crit_N`     | 150      | mg/L              |

## Section 6 — PAA (par 75–80)

| par | Symbol         | Value    | Units            |
|-----|----------------|----------|------------------|
| 75  | `PAA_c`        | passed in | mg/L            |
| 76  | `Y_PAA_P`      | 187.5    | mg/g P           |
| 77  | `Y_PAA_X`      | 45.0     | mg/g X           | (37.5 × 1.2 in source) |
| 78  | `m_PAA`        | 1.05     | g/g P/h          |
| 79  | `X_crit_PAA`   | 2400     | mg/L             |
| 80  | `P_crit_PAA`   | 200      | mg/L             |

## Section 7 — k4 polynomial (par 81–85) — temperature/pH effect on hydrolysis

`k4 = exp(B_1 + B_2·pH + B_3·T + B_4·pH² + B_5·T²)` (line 277 / 299 of indpensim_ode.m)

| par | Symbol | Value      |
|-----|--------|------------|
| 81  | `B_1`  | -64.29     |
| 82  | `B_2`  | -1.825     |
| 83  | `B_3`  | 0.3649     |
| 84  | `B_4`  | 0.1280     |
| 85  | `B_5`  | -4.9496e-4 |

## Section 8 — CO₂, viscosity, evaporation (par 86–95)

| par | Symbol         | Value     | Units            | Notes |
|-----|----------------|-----------|------------------|-------|
| 86  | `delta_c_0`    | 0.89      | —                | CO₂ kla scaling. **Naming inconsistency: source declares `delta_c_o`, ODE reads `delta_c_0`. They are the same parameter (the `o` is a typo for zero) — preserve in port as a single `delta_c_0`.** |
| 87  | `k3`           | 0.005     | —                | Viscosity decline coefficient (water injection). **Source declares `k_3`, ODE reads `k3` — same param.** |
| 88  | `k1`           | 0.001     | —                | Viscosity rise sigmoid steepness #1 |
| 89  | `k2`           | 0.0001    | —                | Viscosity rise sigmoid steepness #2 |
| 90  | `t1`           | 1         | h                | Viscosity rise sigmoid time #1 |
| 91  | `t2`           | 250       | h                | Viscosity rise sigmoid time #2 |
| 92  | `q_co2`        | 0.1353    | mg CO₂/g X/h     | (0.123 × 1.1 in source) |
| 93  | `X_crit_CO2`   | 7570      | mg/L             | CO₂ inhibition threshold |
| 94  | `alpha_evp`    | 5.24e-4   | L/h              | Evaporation rate prefactor |
| 95  | `beta_T`       | 2.88      | —                | Heat-removal exponent (Fc^(beta_T+1) term) |

## Section 9 — Fluid densities & inlet gas (par 96–105)

| par | Symbol     | Value   | Units            |
|-----|------------|---------|------------------|
| 96  | `pho_g`    | 1540    | kg/m³            | Sugar/glucose density (typo preserved: `pho` = `rho`) |
| 97  | `pho_oil`  | 900     | kg/m³            |
| 98  | `pho_w`    | 1000    | kg/m³            | Water |
| 99  | `pho_paa`  | 1000    | kg/m³            | PAA |
| 100 | `O_2_in`   | 0.21    | mole fraction    | Inlet air O₂; +disturbance |
| 101 | `N2_in`    | 0.79    | mole fraction    | Inlet air N₂ |
| 102 | `C_CO2_in` | 0.033   | mole fraction (%)| Inlet CO₂ — value is 0.033%, not 33%. |
| 103 | `Tv`       | 373     | K                | Boiling point of water (evaporation model) |
| 104 | `T0`       | 273     | K                | Reference T |
| 105 | `alpha_1`  | 2451.8  | kJ/m³            | Heat-removal correlation prefactor |

## Naming inconsistencies between Parameter_list.m and indpensim_ode.m

Port these as the **canonical Python names listed in the right column**, regardless of source casing/typo:

| In `Parameter_list.m` | In `indpensim_ode.m` | Canonical Python | par index |
|-----------------------|----------------------|-------------------|-----------|
| `Y_sx`                | `Y_sX`               | `Y_sX`            | 23 |
| `delta_c_o`           | `delta_c_0`          | `delta_c_0`       | 86 |
| `k_3`                 | `k3`                 | `k3`              | 87 |
| `dealta_H_evap`       | `dealta_H_evap`      | `delta_H_evap`    | 54 (fix the typo) |
| `pho_*`               | `pho_*`              | `rho_*`           | 96–99 (fix the typo) |

## Runtime mutations (gotcha refresher)

These five parameters are *modified inside the ODE function* depending on flags:

1. **`mu_p`, `mux_max`, `c_s`, `c_oil`, `abc`, `PAA_c`, `Tcin`, `O_2_in`** all get `+ disturbance` added when `inp1(17) == 1` (lines 211–220 of indpensim_ode.m). The original `par(...)` value is overwritten in a local copy. In Python, do this at the top of `rhs()` with explicit local variables — never mutate the parameter struct.
2. **`mu_h`** (par 20) gets fully overwritten under inhib_flag=1 or 2 (lines 250, 278, 300). Don't read par[19] downstream once inhib_flag != 0; recompute.
3. **`K_diff`** (par 11) is age-modulated per step: `K_diff = par(11) - A_t1*beta_1`, clipped at `K_diff_L`. Treat as a runtime-computed local, not a parameter.

## Suggested Python `io/parameters.py` shape

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class KineticParameters:
    """par(1..22) — penicillin & biomass kinetics."""
    mu_p: float                   # par 1 — from initial state
    mux_max: float                # par 2 — from initial state
    ratio_mu_e_mu_b: float = 0.4  # par 3
    P_std_dev: float = 0.0015     # par 4
    # ... etc
    mu_h: float = 0.003           # par 20 — runtime-overwritten under inhibition

@dataclass(frozen=True)
class ProcessParameters:
    """par(23..46) — stoichiometry, gas transfer, oxygen."""
    Y_sX: float = 1.85
    Y_sP: float = 0.9
    # ... etc
    alpha_kla: float = ...        # par 31 — passed in (no default)
    kla_a: float = 0.38           # par 32 — renamed from `a` to avoid collision
    kla_b: float = 0.34           # par 33 — renamed from `b` to avoid collision
    kla_c: float = -0.38          # par 34
    kla_d: float = 0.25           # par 35

@dataclass(frozen=True)
class ThermalParameters:    # par 47..61, 94..95, 103..105
    ...

@dataclass(frozen=True)
class PHIonParameters:      # par 62..67
    ...

@dataclass(frozen=True)
class NitrogenParameters:   # par 68..74
    ...

@dataclass(frozen=True)
class PAAParameters:        # par 75..80
    ...

@dataclass(frozen=True)
class K4PolynomialParameters:  # par 81..85
    ...

@dataclass(frozen=True)
class CO2ViscosityParameters:  # par 86..93
    ...

@dataclass(frozen=True)
class DensityParameters:    # par 96..102
    rho_g: float = 1540.0
    rho_oil: float = 900.0
    rho_w: float = 1000.0
    rho_paa: float = 1000.0
    O_2_in: float = 0.21
    N2_in: float = 0.79
    C_CO2_in: float = 0.033

@dataclass(frozen=True)
class Parameters:
    """The full 105-parameter set, grouped semantically."""
    kinetics: KineticParameters
    process: ProcessParameters
    thermal: ThermalParameters
    ph_ion: PHIonParameters
    nitrogen: NitrogenParameters
    paa: PAAParameters
    k4: K4PolynomialParameters
    co2_visc: CO2ViscosityParameters
    density: DensityParameters

    @classmethod
    def default(cls, mu_p: float, mux_max: float, alpha_kla: float,
                N_conc_paa: float, PAA_c: float) -> "Parameters":
        """Mirrors Parameter_list(x0, alpha_kla, N_conc_paa, PAA_c)."""
        ...

    def to_legacy_par_vector(self) -> "np.ndarray":
        """Return a 105-element float array in the original par(1..105) order.
        Useful for line-by-line ODE port that mirrors the MATLAB indexing.
        """
        ...
```

The `to_legacy_par_vector()` method is the bridge: keeps the line-by-line ODE port readable (`par[19]` matches `par(20)` minus 1), while letting the rest of the Python codebase use named struct fields.

## Open question

**Section 8 unit on `delta_c_0`** is dimensionless in source comments but enters the dissolved-CO₂ ODE as a kla scaling factor (`kla*delta_c_0*(C_star_CO2 - y(29))`). Likely a kla-scale-down ratio (CO₂ is more soluble than O₂). Verify against MATLAB output during validation.
