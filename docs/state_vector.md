# 33-state vector glossary (`y` in `indpensim_ode.m`)

Canonical reference for the state vector integrated by the ODE solver. All MATLAB indices are 1-based; Python (numpy) indices are 0-based. The "X.field" column is the field name used in the output struct (set in `indpensim.m:249–319`) and therefore the column name in the dumped reference CSV.

| MATLAB | Python | X.field        | Symbol | Description | Units |
|--------|--------|----------------|--------|-------------|-------|
| y(1)   | y[0]   | S              | S       | Substrate (sugar) concentration | g/L |
| y(2)   | y[1]   | DO2            | DO₂     | Dissolved oxygen | mg/L |
| y(3)   | y[2]   | O2             | O₂_off  | Oxygen off-gas | mole fraction |
| y(4)   | y[3]   | P              | P       | Penicillin concentration | g/L |
| y(5)   | y[4]   | V              | V       | Broth volume | L |
| y(6)   | y[5]   | Wt             | Wt      | Total vessel weight | kg |
| y(7)   | y[6]   | pH             | [H⁺]    | **DURING integration: [H⁺] in mol/L** (despite field name). After the integration loop, `indpensim.m:382` converts the whole trajectory in-place: `X.pH.y = -log(X.pH.y)/log(10)`. The dumped reference CSV therefore stores pH (4–10), not [H⁺]. Python port: keep [H⁺] inside `rhs()`, convert to pH only at output. See gotcha #2. | mol/L (internal) / pH (output) |
| y(8)   | y[7]   | T              | T       | Temperature | K |
| y(9)   | y[8]   | Q              | Q       | Cumulative generated heat | kcal |
| y(10)  | y[9]   | Viscosity      | μ       | Viscosity (overridable via Ctrl_flags.Vis=1) | cP |
| y(11)  | y[10]  | Culture_age    | ∫X dt   | Integral of total biomass (A0+A1+A3+A4) | g·h/L |
| y(12)  | y[11]  | a0             | A0      | Growing biomass region (branching) | g/L |
| y(13)  | y[12]  | a1             | A1      | Non-growing biomass region (extension) | g/L |
| y(14)  | y[13]  | a3             | A3      | Degenerated biomass region | g/L |
| y(15)  | y[14]  | a4             | A4      | Autolysed biomass region | g/L |
| y(16)  | y[15]  | n0             | n₀      | Vacuole number density, size bin 0 (boundary cell) | # cm⁻¹ L⁻¹ |
| y(17)  | y[16]  | n1             | n₁      | Vacuole bin 1 (interior FD) | # cm⁻¹ L⁻¹ |
| y(18)  | y[17]  | n2             | n₂      | Vacuole bin 2 | # cm⁻¹ L⁻¹ |
| y(19)  | y[18]  | n3             | n₃      | Vacuole bin 3 | # cm⁻¹ L⁻¹ |
| y(20)  | y[19]  | n4             | n₄      | Vacuole bin 4 | # cm⁻¹ L⁻¹ |
| y(21)  | y[20]  | n5             | n₅      | Vacuole bin 5 | # cm⁻¹ L⁻¹ |
| y(22)  | y[21]  | n6             | n₆      | Vacuole bin 6 | # cm⁻¹ L⁻¹ |
| y(23)  | y[22]  | n7             | n₇      | Vacuole bin 7 | # cm⁻¹ L⁻¹ |
| y(24)  | y[23]  | n8             | n₈      | Vacuole bin 8 | # cm⁻¹ L⁻¹ |
| y(25)  | y[24]  | n9             | n₉      | Vacuole bin 9 (last interior) | # cm⁻¹ L⁻¹ |
| y(26)  | y[25]  | nm             | n_max   | Maximum-size vacuole department (lumped pool) | # L⁻¹ |
| y(27)  | y[26]  | phi0           | φ₀      | Mean vacuole volume (age-zero) | mL/L |
| y(28)  | y[27]  | CO2outgas      | CO₂_off | CO₂ off-gas concentration | % |
| y(29)  | y[28]  | CO2_d          | CO₂_d   | Dissolved CO₂ | g/L |
| y(30)  | y[29]  | PAA            | PAA     | Phenylacetic acid | mg/L |
| y(31)  | y[30]  | NH3            | NH₃     | Nitrogen (ammonia) | mg/L |
| y(32)  | y[31]  | mu_P_calc      | ∫μ_p dt | Integral of penicillin growth-rate parameter (= mu_p·t since mu_p constant). Header docs mis-label as "current growth rate". | dimensionless |
| y(33)  | y[32]  | mu_X_calc      | ∫μ_e dt | Integral of effective biomass growth rate. Diagnostic; doesn't feed back. | dimensionless |

## Critical porting gotchas

### 1. `dy = zeros(31, 1)` allocates only 31 — but the function writes 33
Lines 445 / 569–570: MATLAB silently grows the array when `dy(32)`, `dy(33)` are written. **Python must allocate `np.zeros(33)` up front.** A `np.zeros(31)` followed by `dy[31] = ...` would IndexError.

### 2. `y(7)` is mutated mid-function in the basic-pH branch
Line 490: `y(7) = (1e-14/y(7) - y(7));` switches the [H⁺] balance to an [OH⁻] balance and *overwrites the local copy*. In MATLAB, function args are pass-by-value, so this only mutates the local `y` — the integrator's state is unaffected. **In Python, `y` is a numpy array passed by reference. Must copy first:**

```python
def rhs(t, y, ...):
    y = y.copy()   # MANDATORY — avoid mutating solver state
    ...
```

Skipping this corrupts the integrator's internal step buffers and causes nondeterministic divergence.

### 3. `mu_h` is not a parameter — it's recomputed per step
`par(20)` provides an initial value, but every branch of the `inhib_flag` block (lines 250, 278, 300) overwrites it. Don't treat `mu_h` as a static parameter in Python; recompute inside the RHS as the MATLAB does.

### 4. Three different `b`'s in the codebase
- `b` in `indpensim_ode.m`: scalar parameter `par(33)` — kla correlation exponent.
- `b` in `Raman_Sim.m`: local scalar/loop variable, reused multiple times.
- `b` in `PAA_PLS_model.mat`: 10×212 PLS regression coefficient matrix.

Keep namespaced in Python: `kla_b`, `pls.b`, etc.

### 5. Vacuole PDE uses `y(n)` indexed via a running counter
Lines 332–387: a Python-friendly rewrite is `y[15:25]` for the vacuole bins; the central-difference stencil `(y[i+1] − 2*y[i] + y[i−1]) / Δr²` becomes vectorisable as `np.diff(y[15:26], n=2) / dr**2` and `(y[16:25] − y[14:23]) / (2*dr)` for the advection term.

### 6. `inhib_flag = 0` is a "free run" mode
Sets every inhibition multiplier to 1.0 and `mu_h = 0.003`. Useful for unit-testing kinetics in isolation — port this branch first, since it gives the simplest reference for cross-checking biomass/penicillin growth.

### 7. `viscosity(viscosity<4) = 1` and `viscosity(viscosity<=4) = 1` — both live
Lines 242 and 457 — same idiom with `<` vs `<=`. **Both are live.** Line 194 may reassign `viscosity = inp1(12)` (recorded value) when `Ctrl_flags.Vis=1` — that happens *after* the line-242 floor but *before* the line-457 floor. Skipping either changes behaviour on the recorded-viscosity path. Port both.

### 8. `dy(11) = y(12)+y(13)+y(14)+y(15)` is *not* a rate of change
It's the *value* being treated as a rate — i.e., y(11) integrates the instantaneous total biomass, ending up with units of g·h/L. This appears intentional (used downstream as a cumulative "biomass exposure" metric).

## Inputs vector (`inp1`, 26 elements) — controller→ODE handoff

| MATLAB | Python | Name | Units |
|--------|--------|------|-------|
| inp1(1)  | inp1[0]  | inhibition flag (0/1/2) | — |
| inp1(2)  | inp1[1]  | Fs sugar feed | L/h |
| inp1(3)  | inp1[2]  | Fg aeration | m³/h (converted to /60 inside) |
| inp1(4)  | inp1[3]  | RPM | rpm |
| inp1(5)  | inp1[4]  | Fc cooling water | L/h |
| inp1(6)  | inp1[5]  | Fh heating water | L/h |
| inp1(7)  | inp1[6]  | Fb base | L/h |
| inp1(8)  | inp1[7]  | Fa acid | L/h |
| inp1(9)  | inp1[8]  | h_ode step size | h |
| inp1(10) | inp1[9]  | Fw water for injection | L/h |
| inp1(11) | inp1[10] | head pressure | bar |
| inp1(12) | inp1[11] | recorded viscosity | cP |
| inp1(13) | inp1[12] | F_discharge | L/h |
| inp1(14) | inp1[13] | F_PAA | L/h |
| inp1(15) | inp1[14] | F_oil | L/h |
| inp1(16) | inp1[15] | NH₃ shots | kg |
| inp1(17) | inp1[16] | disturbance flag | — |
| inp1(18) | inp1[17] | dmuP | h⁻¹ |
| inp1(19) | inp1[18] | dmuX | h⁻¹ |
| inp1(20) | inp1[19] | distcs sugar inlet | g/L |
| inp1(21) | inp1[20] | distcoil oil inlet | g/L |
| inp1(22) | inp1[21] | distabc acid/base | mol/L |
| inp1(23) | inp1[22] | distPAA | mg/L |
| inp1(24) | inp1[23] | distTcin | K |
| inp1(25) | inp1[24] | distO₂_in | mg/L |
| inp1(26) | inp1[25] | viscosity flag (0=sim, 1=recorded) | — |

## Suggested Python `ode/state.py`

```python
from dataclasses import dataclass
from enum import IntEnum


class S(IntEnum):
    """0-based indices into the 33-element state vector."""
    SUBSTRATE = 0          # g/L
    DO2 = 1                # mg/L
    O2_OFF = 2             # mole fraction
    PENICILLIN = 3         # g/L
    VOLUME = 4             # L
    WEIGHT = 5             # kg
    H_PLUS = 6             # mol/L  (NOT pH — pH = -log10(y[6]))
    TEMPERATURE = 7        # K
    Q_HEAT = 8             # kcal (cumulative)
    VISCOSITY = 9          # cP
    INTEGRAL_X = 10        # g·h/L
    A0 = 11                # g/L  growing
    A1 = 12                # g/L  non-growing
    A3 = 13                # g/L  degenerated
    A4 = 14                # g/L  autolysed
    # Vacuole interior bins n_0..n_9 → indices 15..24 (use VAC_BIN_SLICE; not enumerated to avoid off-by-one bait)
    N_VAC_MAX = 25         # max-size department
    PHI_0 = 26             # mean vacuole volume
    CO2_OFF = 27           # %
    CO2_DISSOLVED = 28     # g/L
    PAA = 29               # mg/L
    NH3 = 30               # mg/L
    INTEGRAL_MU_P = 31     # ∫mu_p dt
    INTEGRAL_MU_E = 32     # ∫mu_e dt


N_STATES = 33
N_VAC_BINS = 10  # interior bins n_0 .. n_9 → indices 15..24
VAC_BIN_SLICE = slice(15, 25)


def initial_state() -> "np.ndarray":
    """Return a 33-element zero-initialised state vector."""
    import numpy as np
    return np.zeros(N_STATES, dtype=float)
```
