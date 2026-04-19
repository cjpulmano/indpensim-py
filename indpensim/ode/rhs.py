"""Right-hand side of the IndPenSim 33-state ODE — line-by-line port of indpensim_ode.m.

This is a single-function port by design (see plan amendment): the original
578-line MATLAB function decomposes into kinetics / heat / gas-transfer /
vacuole-PDE *blocks*, but those blocks share intermediate quantities so
heavily that splitting into separate testable units would yield only stub-
consistency tests, not physics tests. The rule is: port as one function,
validate end-to-end against the MATLAB reference, then refactor with the
passing test as a regression gate.

Line references in comments (e.g. "MATLAB:248") map to indpensim_ode.m so
this file can be audited side-by-side with the original.

Critical gotchas (see docs/state_vector.md):
  - y is COPIED at the top — line 490 of MATLAB mutates y(7), which would
    corrupt the solver state if we did it in-place on the numpy array.
  - dy is allocated as 33 elements; MATLAB allocates 31 then silently grows
    when writing dy(32) and dy(33).
  - mu_h (par[19]) is overwritten under inhib_flag != 0 — it's effectively
    a runtime-computed quantity, not a parameter.
  - K_diff (par[10]) is age-modulated per step.
  - Disturbances override parameter values via local copies, never the
    par array itself.

Signature is suitable for scipy.integrate.solve_ivp:
    sol = solve_ivp(rhs, t_span, y0, args=(inp1, par_vec), method='BDF', ...)
"""
from __future__ import annotations

import numpy as np


N_STATES: int = 33


def rhs(t: float, y: np.ndarray, inp1: np.ndarray, par: np.ndarray) -> np.ndarray:
    """Compute dy/dt at (t, y) given controller inputs `inp1` and params `par`.

    Args:
        t: time [h]
        y: state vector, shape (33,). 0-based index = MATLAB index − 1.
        inp1: 26-element controller input vector. See docs/state_vector.md
            for the channel mapping.
        par: 105-element legacy parameter vector. Build via
            ``Parameters.default(...).to_legacy_par_vector()``.

    Returns:
        dy: shape (33,) derivative vector.
    """
    # MUST copy — MATLAB:490 mutates y(7) on the basic-pH branch. Pass-by-value
    # in MATLAB is harmless; numpy is pass-by-reference and would corrupt the
    # integrator's internal step buffers. (Gotcha #2.)
    y = np.asarray(y, dtype=np.float64).copy()

    # --- Parameter unpacking (MATLAB:67-173) -------------------------------
    # Group 1 — kinetics, par(1..22)
    mu_p             = par[0]    # par(1)
    mux_max          = par[1]    # par(2)
    ratio_mu_e_mu_b  = par[2]
    P_std_dev        = par[3]
    mean_P           = par[4]
    mu_v             = par[5]
    mu_a             = par[6]
    mu_diff          = par[7]
    beta_1           = par[8]
    K_b              = par[9]
    K_diff_init      = par[10]   # NOTE: age-modulated below; do not reuse name
    K_diff_L         = par[11]
    K_e              = par[12]
    K_v              = par[13]
    delta_r          = par[14]
    k_v              = par[15]
    D                = par[16]
    rho_a0           = par[17]
    rho_d            = par[18]
    mu_h             = par[19]   # gotcha #3 — overwritten under inhib_flag != 0
    r_0              = par[20]
    delta_0          = par[21]

    # Group 2 — process, par(23..46)
    Y_sX             = par[22]
    Y_sP             = par[23]
    m_s              = par[24]
    c_oil            = par[25]   # +disturbance
    c_s              = par[26]   # +disturbance
    Y_O2_X           = par[27]
    Y_O2_P           = par[28]
    m_O2_X           = par[29]
    alpha_kla        = par[30]
    kla_a            = par[31]   # par(32) — was `a` in source
    kla_b            = par[32]   # par(33) — was `b` in source
    kla_c            = par[33]
    kla_d            = par[34]
    Henrys_c         = par[35]
    n_imp            = par[36]
    r                = par[37]
    r_imp            = par[38]
    Po               = par[39]
    epsilon          = par[40]
    # g (par 42), R (par 43) — only R is used downstream
    R                = par[42]
    X_crit_DO2       = par[43]
    P_crit_DO2       = par[44]
    A_inhib          = par[45]

    # Group 3 — thermal, par(47..61)
    Tf               = par[46]
    Tw               = par[47]
    Tcin             = par[48]   # +disturbance
    Th               = par[49]
    Tair             = par[50]
    C_ps             = par[51]
    C_pw             = par[52]
    delta_H_evap     = par[53]   # was "dealta_H_evap" in source
    U_jacket         = par[54]
    A_c              = par[55]
    Eg               = par[56]
    Ed               = par[57]
    k_g              = par[58]
    k_d              = par[59]
    Y_QX             = par[60]

    # Group 4 — pH/ions, par(62..67)
    abc              = par[61]   # +disturbance
    gamma1           = par[62]
    gamma2           = par[63]
    m_ph             = par[64]
    K1               = par[65]
    K2               = par[66]

    # Group 5 — nitrogen, par(68..74)
    N_conc_oil       = par[67]
    N_conc_paa       = par[68]
    N_conc_shot      = par[69]
    Y_NX             = par[70]
    Y_NP             = par[71]
    m_N              = par[72]
    X_crit_N         = par[73]

    # Group 6 — PAA, par(75..80)
    PAA_c            = par[74]   # +disturbance
    Y_PAA_P          = par[75]
    Y_PAA_X          = par[76]
    m_PAA            = par[77]
    X_crit_PAA       = par[78]
    P_crit_PAA       = par[79]

    # Group 7 — k4 polynomial, par(81..85)
    B_1              = par[80]
    B_2              = par[81]
    B_3              = par[82]
    B_4              = par[83]
    B_5              = par[84]

    # Group 8 — CO2/viscosity, par(86..93)
    delta_c_0        = par[85]
    k3               = par[86]
    k1               = par[87]
    k2               = par[88]
    t1               = par[89]
    t2               = par[90]
    q_co2            = par[91]
    X_crit_CO2       = par[92]

    # Group 8b — alpha_evp/beta_T, par(94..95)
    alpha_evp        = par[93]
    beta_T           = par[94]

    # Group 9 — densities & inlet gas, par(96..102)
    rho_g            = par[95]   # was pho_g
    rho_oil          = par[96]
    rho_w            = par[97]
    rho_paa          = par[98]
    O_2_in           = par[99]   # +disturbance
    N2_in            = par[100]
    C_CO2_in         = par[101]

    # Group 9b — Tv, T0, alpha_1, par(103..105)
    Tv               = par[102]
    T0               = par[103]
    alpha_1          = par[104]

    # --- Process inputs (MATLAB:176-208) ----------------------------------
    inhib_flag = int(inp1[0])
    Fs   = inp1[1]
    Fg   = inp1[2] / 60.0      # MATLAB:178 — convert m^3/h to m^3/min
    RPM  = inp1[3]
    Fc   = inp1[4]
    Fh   = inp1[5]
    Fb   = inp1[6]
    Fa   = inp1[7]
    step1 = inp1[8]            # ode step size [h]
    Fw   = max(inp1[9], 0.0)   # MATLAB:188 — clip negative water flow to 0
    pressure = inp1[10]
    # Viscosity flag — MATLAB:191-195
    if int(inp1[25]) == 0:     # Ctrl_flags.Vis = 0 → use simulated viscosity
        viscosity = y[9]
    else:                      # Ctrl_flags.Vis = 1 → use recorded viscosity
        viscosity = inp1[11]
    F_discharge = inp1[12]
    Fpaa = inp1[13]
    Foil = inp1[14]
    NH3_shots = inp1[15]
    dist_flag = int(inp1[16])
    distMuP    = inp1[17]
    distMuX    = inp1[18]
    distsc     = inp1[19]
    distcoil   = inp1[20]
    distabc    = inp1[21]
    distPAA    = inp1[22]
    distTcin   = inp1[23]
    distO_2_in = inp1[24]

    # MATLAB:209 — broth density (g/L based; mixes biomass regions in)
    pho_b = 1100.0 + y[3] + y[11] + y[12] + y[13] + y[14]

    # MATLAB:211-220 — disturbance overrides (mutate locals, never par)
    if dist_flag == 1:
        mu_p    = mu_p    + distMuP
        mux_max = mux_max + distMuX
        c_s     = c_s     + distsc
        c_oil   = c_oil   + distcoil
        abc     = abc     + distabc
        PAA_c   = PAA_c   + distPAA
        Tcin    = Tcin    + distTcin
        O_2_in  = O_2_in  + distO_2_in

    # --- Process intermediates (MATLAB:223-243) ---------------------------
    # Age-dependent biomass term (MATLAB:225)
    A_t1 = y[10] / (y[11] + y[12] + y[13] + y[14])

    s        = y[0]                          # substrate
    a_1      = y[12]                         # non-growing region (extension)
    a_0      = y[11]                         # branching region
    a_3      = y[13]
    total_X  = y[11] + y[12] + y[13] + y[14] # MATLAB:231

    # Vessel hydrostatics (MATLAB:233-240)
    h_b = (y[4] / 1000.0) / (np.pi * (r ** 2))
    h_b = h_b * (1.0 - epsilon)
    pressure_bottom = 1.0 + pressure + (pho_b * h_b) * 9.81 * 1e-5
    pressure_top    = 1.0 + pressure
    log_mean_pressure = (pressure_bottom - pressure_top) / np.log(pressure_bottom / pressure_top)
    total_pressure = log_mean_pressure

    # Viscosity floor (MATLAB:242 — gotcha #7, both this and 457 are live)
    if viscosity < 4:
        viscosity = 1.0

    DOstar_tp = (total_pressure * O_2_in) / Henrys_c   # mg/L

    # --- Inhibition flag dispatch (MATLAB:245-301) ------------------------
    if inhib_flag == 0:
        pH_inhib = 1.0
        NH3_inhib = 1.0
        T_inhib = 1.0
        mu_h = 0.003
        DO_2_inhib_X = 1.0
        DO_2_inhib_P = 1.0
        CO2_inhib = 1.0
        PAA_inhib_X = 1.0
        PAA_inhib_P = 1.0

    elif inhib_flag == 1:
        # MATLAB:258-279 — DO2, T, pH inhibition (no PAA, NH3, CO2)
        pH_inhib   = 1.0 / (1.0 + (y[6] / K1) + (K2 / y[6]))
        NH3_inhib  = 1.0
        # MATLAB:264 — note the `*0+1` that nullifies the Arrhenius expression
        T_inhib    = (k_g * np.exp(-(Eg / (R * y[7]))) - k_d * np.exp(-(Ed / (R * y[7])))) * 0.0 + 1.0
        CO2_inhib  = 1.0
        DO_2_inhib_X = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        DO_2_inhib_P = 0.5 * (1.0 - np.tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        PAA_inhib_X = 1.0
        PAA_inhib_P = 1.0
        # T·pH effect on hydrolysis (MATLAB:276-278)
        pH_val = -np.log10(y[6])
        k4 = np.exp((B_1 + B_2 * pH_val + B_3 * y[7] + B_4 * (pH_val ** 2)) + B_5 * (y[7] ** 2))
        mu_h = k4

    elif inhib_flag == 2:
        # MATLAB:282-301 — full inhibition set
        pH_inhib   = 1.0 / (1.0 + (y[6] / K1) + (K2 / y[6]))
        NH3_inhib  = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_N - y[30])))
        T_inhib    = k_g * np.exp(-(Eg / (R * y[7]))) - k_d * np.exp(-(Ed / (R * y[7])))
        CO2_inhib  = 0.5 * (1.0 + np.tanh(A_inhib * (X_crit_CO2 - y[28] * 1000.0)))
        DO_2_inhib_X = 0.5 * (1.0 - np.tanh(A_inhib * (X_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        DO_2_inhib_P = 0.5 * (1.0 - np.tanh(A_inhib * (P_crit_DO2 * ((total_pressure * O_2_in) / Henrys_c) - y[1])))
        PAA_inhib_X = 0.5 * (1.0 + np.tanh(X_crit_PAA - y[29]))
        PAA_inhib_P = 0.5 * (1.0 + np.tanh(-P_crit_PAA + y[29]))
        pH_val = -np.log10(y[6])
        k4 = np.exp((B_1 + B_2 * pH_val + B_3 * y[7] + B_4 * (pH_val ** 2)) + B_5 * (y[7] ** 2))
        mu_h = k4
    else:
        raise ValueError(f"inhib_flag must be 0, 1, or 2; got {inhib_flag}")

    # --- Kinetic rate equations (MATLAB:307-351) --------------------------
    # Penicillin inhibition curve (Gaussian on substrate)
    P_inhib = 2.5 * P_std_dev * (
        (P_std_dev * np.sqrt(2.0 * np.pi)) ** -1
        * np.exp(-0.5 * ((s - mean_P) / P_std_dev) ** 2)
    )
    mu_a0 = ratio_mu_e_mu_b * mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X
    mu_e  =                   mux_max * pH_inhib * NH3_inhib * T_inhib * DO_2_inhib_X * CO2_inhib * PAA_inhib_X

    # Age-modulated K_diff (MATLAB:314-317)
    K_diff = K_diff_init - (A_t1 * beta_1)
    if K_diff < K_diff_L:
        K_diff = K_diff_L

    # Branching A_0 region (MATLAB:319-320)
    r_b0  = mu_a0 * a_1 * s / (K_b + s)
    r_sb0 = Y_sX * r_b0

    # Extension A_1 region (MATLAB:323-324)
    r_e1  = (mu_e * a_0 * s) / (K_e + s)
    r_se1 = Y_sX * r_e1

    # Differentiation A_0 → A_1 (MATLAB:327-328)
    r_d1 = mu_diff * a_0 / (K_diff + s)
    r_m0 = m_s * a_0 / (K_diff + s)

    # Vacuole volume accumulator (MATLAB:332-340)
    # phi(1)=y(27), then for k=2..10, r_mean(k) = 1.5e-4 + (k-2)*delta_r,
    #                                phi(k)    = (4*pi*r_mean(k)^3)/3 * y(n) * delta_r,
    # where n iterates 17..25 (MATLAB 1-based) → Python indices 16..24.
    # In Python: y[15] is n_0, y[16..24] are n_1..n_9 (per state.py).
    # The MATLAB loop starts at k=2 and reads y(17) which is n_1 — i.e. y[16] in Python.
    phi_acc = y[26]   # phi(1) = y(27) → y[26]
    for k in range(2, 11):
        r_mean_k = 1.5e-4 + (k - 2) * delta_r
        n_index = 16 + (k - 2)   # k=2 → 16, ..., k=10 → 24
        phi_acc = phi_acc + ((4.0 * np.pi * r_mean_k ** 3) / 3.0) * y[n_index] * delta_r
    v_2 = phi_acc

    # Density of non-growing region (MATLAB:341-342)
    rho_a1 = a_1 / ((a_1 / rho_a0) + v_2)
    v_a1   = a_1 / (2.0 * rho_a1) - v_2

    # Penicillin produced from non-growing region (MATLAB:344)
    r_p = mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P - mu_h * y[3]

    # Vacuole formation (MATLAB:348)
    r_m1 = (m_s * rho_a0 * v_a1 * s) / (K_v + s)

    # Biomass autolysis (MATLAB:351)
    r_d4 = mu_a * a_3

    # --- Vacuole PDE finite-difference block (MATLAB:354-389) -------------
    # n_0 boundary cell (MATLAB:356)
    dn0_dt = ((mu_v * v_a1) / (K_v + s)) * ((6.0 / np.pi) * ((r_0 + delta_0) ** -3)) - k_v * y[15]

    # n_1..n_9 interior cells (MATLAB:358-378). Vectorised: same stencil
    # for each cell using central differences on the size axis.
    # MATLAB loop runs n=17..25 (i.e., interior cells y(17)..y(25));
    # neighbours at n-1, n+1 → y(16)..y(24) and y(18)..y(26).
    # Python: cells y[16..24], neighbours y[15..23] and y[17..25].
    interior     = y[16:25]   # n_1..n_9, length 9
    neighbour_lo = y[15:24]   # n_0..n_8
    neighbour_hi = y[17:26]   # n_2..n_10  (n_10 = n_max = y[25])
    dn_interior = (
        -k_v * (neighbour_hi - neighbour_lo) / (2.0 * delta_r)
        + D   * (neighbour_hi - 2.0 * interior + neighbour_lo) / (delta_r ** 2)
    )
    # n_k for the max-vacuole department equation (MATLAB:378, 386).
    # MATLAB sequence:
    #   line 378: n_k = dn9_dt     (derivative)
    #   line 386: dn_m_dt = k_v*n_k/(r_m-r_k) - mu_a*y(26)    -- uses n_k=dn9_dt
    #   line 387: n_k = y(25)     (reassigned to a state for use at line 407)
    # So at the moment dn_m_dt is computed, n_k IS dn9_dt — not y(25).
    # dn_interior[-1] is dn9_dt by construction.
    n_k_for_max_dept = dn_interior[-1]
    k_idx_low  = 10            # MATLAB:379 — k = 10
    r_k = r_0 + (k_idx_low - 2) * delta_r
    k_idx_high = 12            # MATLAB:383 — k = 12
    r_m = r_0 + (k_idx_high - 2) * delta_r

    dn_m_dt   = k_v * n_k_for_max_dept / (r_m - r_k) - mu_a * y[25]
    # Mean vacuole volume rate (MATLAB:389)
    dphi_0_dt = ((mu_v * v_a1) / (K_v + s)) - k_v * y[15] * (np.pi * (r_0 + delta_0) ** 3) / 6.0

    # --- Volume / weight (MATLAB:391-401) ---------------------------------
    F_evp = y[4] * alpha_evp * (np.exp(2.5 * (y[7] - T0) / (Tv - T0)) - 1.0)
    pho_feed = (c_s / 1000.0) * rho_g + (1.0 - c_s / 1000.0) * rho_w
    dilution = Fs + Fb + Fa + Fw - F_evp + Fpaa
    dV1 = Fs + Fb + Fa + Fw + F_discharge / (pho_b / 1000.0) - F_evp + Fpaa
    dWt = (
        Fs * pho_feed / 1000.0
        + rho_oil / 1000.0 * Foil
        + Fb + Fa + Fw + F_discharge - F_evp
        + Fpaa * rho_paa / 1000.0
    )

    # --- Biomass region ODEs (MATLAB:403-411) -----------------------------
    # n_k for the boundary flux into a_3 (MATLAB:407 reads n_k from line 387 = y(25))
    n_k_boundary = y[24]   # NB: the MATLAB code uses the latest assigned n_k
    da_0_dt = r_b0 - r_d1 - y[11] * dilution / y[4]
    da_1_dt = (
        r_e1 - r_b0 + r_d1
        - (np.pi * ((r_k + r_m) ** 3) / 6.0) * rho_d * k_v * n_k_boundary
        - y[12] * dilution / y[4]
    )
    da_3_dt = (np.pi * ((r_k + r_m) ** 3) / 6.0) * rho_d * k_v * n_k_boundary - r_d4 - y[13] * dilution / y[4]
    da_4_dt = r_d4 - y[14] * dilution / y[4]

    # Penicillin production (MATLAB:413)
    dP_dt = r_p - y[3] * dilution / y[4]

    # Active biomass rate, total biomass (MATLAB:415-418)
    X_1 = da_0_dt + da_1_dt + da_3_dt + da_4_dt
    X_t = y[11] + y[12] + y[13] + y[14]

    # Reaction heat (MATLAB:421-428)
    Qrxn_X = X_1   * Y_QX * y[4] * Y_O2_X / 1000.0
    Qrxn_P = dP_dt * Y_QX * y[4] * Y_O2_P / 1000.0
    Qrxn_t = Qrxn_X + Qrxn_P
    if Qrxn_t < 0:
        Qrxn_t = 0.0

    # --- Power & gas transfer (MATLAB:432-437) ---------------------------
    N_imp_rps = RPM / 60.0
    D_imp = 2.0 * r_imp
    unaerated_power = n_imp * Po * pho_b * (N_imp_rps ** 3) * (D_imp ** 5)
    P_g = 0.706 * (((unaerated_power ** 2) * N_imp_rps * D_imp ** 3) / (Fg ** 0.56)) ** 0.45
    P_n = P_g / unaerated_power
    variable_power = (n_imp * Po * pho_b * (N_imp_rps ** 3) * (D_imp ** 5) * P_n) / 1000.0  # kW

    # --- Allocate dy (MATLAB:445 allocates 31, then writes dy(32),dy(33);
    # we allocate 33 directly — see gotcha #1) ---------------------------
    dy = np.zeros(N_STATES, dtype=np.float64)

    # --- dy(1) substrate (MATLAB:448) -------------------------------------
    dy[0] = (
        -r_se1 - r_sb0 - r_m0 - r_m1
        - (Y_sP * mu_p * rho_a0 * v_a1 * P_inhib * DO_2_inhib_P * PAA_inhib_P)
        + Fs * c_s / y[4]
        + Foil * c_oil / y[4]
        - y[0] * dilution / y[4]
    )

    # --- dy(2) DO2 (MATLAB:450-463) --------------------------------------
    V_s = Fg / (np.pi * (r ** 2))     # superficial gas velocity m/s
    T_K = y[7]
    V   = y[4]
    V_m = y[4] / 1000.0
    P_air = (V_s * R * T_K * V_m / (22.4 * h_b)) * np.log(
        1.0 + pho_b * 9.81 * h_b / (pressure_top * 1e5)
    )
    P_t1 = variable_power + P_air
    # Second viscosity floor (MATLAB:457 — gotcha #7, both floors are live)
    if viscosity <= 4:
        viscosity = 1.0
    vis_scaled = viscosity / 100.0
    oil_f = Foil / V
    kla = (
        alpha_kla
        * (V_s ** kla_a) * ((P_t1 / V_m) ** kla_b) * (vis_scaled ** kla_c)
        * (1.0 - oil_f ** kla_d)
    )
    OUR = (-X_1) * Y_O2_X - m_O2_X * X_t - dP_dt * Y_O2_P
    OTR = kla * (DOstar_tp - y[1])
    dy[1] = OUR + OTR - (y[1] * dilution / y[4])

    # --- dy(3) O2 off-gas (MATLAB:466-469) -------------------------------
    Vg = epsilon * V_m
    Qfg_in  = 60.0 * Fg * 1000.0 * 32.0 / 22.4
    Qfg_out = 60.0 * Fg * (N2_in / (1.0 - y[2] - y[27] / 100.0)) * 1000.0 * 32.0 / 22.4
    dy[2] = (Qfg_in * O_2_in - Qfg_out * y[2] - 0.001 * OTR * V_m * 60.0) / (Vg * 28.97 * 1000.0 / 22.4)

    # --- dy(4..6) penicillin, volume, weight (MATLAB:472-476) ------------
    dy[3] = r_p - y[3] * dilution / y[4]
    dy[4] = dV1
    dy[5] = dWt

    # --- dy(7) pH (MATLAB:478-504) ---------------------------------------
    pH_dis = Fs + Foil + Fb + Fa + F_discharge + Fw   # water excluded per comment
    if -np.log10(y[6]) < 7.0:
        # acidic — [H+] balance
        cb = -abc
        ca =  abc
        # y[6] unchanged
        pH_balance = 0
    else:
        # basic — [OH-] balance; mutate local copy of y[6] (we copied y at top)
        cb =  abc
        ca = -abc
        y[6] = (1e-14 / y[6] - y[6])
        pH_balance = 1
    B_ion = (y[6] * y[4] + ca * Fa * step1 + cb * Fb * step1) / (y[4] + Fb * step1 + Fa * step1)
    B_ion = -B_ion
    if pH_balance == 1:   # basic branch (MATLAB:496-499)
        dy[6] = (
            -gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X)
            - gamma1 * r_p
            - gamma2 * pH_dis
            + ((-B_ion - np.sqrt(B_ion ** 2 + 4e-14)) / 2.0 - y[6])
        )
    else:                  # acidic branch (MATLAB:500-504)
        dy[6] = (
            +gamma1 * (r_b0 + r_e1 + r_d4 + r_d1 + m_ph * total_X)
            + gamma1 * r_p
            + gamma2 * pH_dis
            + ((-B_ion + np.sqrt(B_ion ** 2 + 4e-14)) / 2.0 - y[6])
        )

    # --- dy(8) temperature, dy(9) Q (MATLAB:506-518) ---------------------
    Ws = P_t1
    Qcon = U_jacket * A_c * (y[7] - Tair)
    dQ_dt = (
        Fs * pho_feed * C_ps * (Tf - y[7]) / 1000.0
        + Fw * rho_w * C_pw * (Tw - y[7]) / 1000.0
        - F_evp * pho_b * C_pw / 1000.0
        - delta_H_evap * F_evp * rho_w / 1000.0
        + Qrxn_t + Ws
        - (alpha_1 / 1000.0) * Fc ** (beta_T + 1.0)
            * ((y[7] - Tcin) / (Fc / 1000.0 + (alpha_1 * (Fc / 1000.0) ** beta_T) / 2.0 * pho_b * C_ps))
        - (alpha_1 / 1000.0) * Fh ** (beta_T + 1.0)
            * ((y[7] - Th)   / (Fh / 1000.0 + (alpha_1 * (Fh / 1000.0) ** beta_T) / 2.0 * pho_b * C_ps))
        - Qcon
    )
    dy[7] = dQ_dt / ((y[4] / 1000.0) * C_pw * pho_b)
    dy[8] = dQ_dt

    # --- dy(10) viscosity (MATLAB:521) -----------------------------------
    dy[9] = (
        3.0 * (a_0 ** (1.0 / 3.0))
        * (1.0 / (1.0 + np.exp(-k1 * (t - t1))))
        * (1.0 / (1.0 + np.exp(-k2 * (t - t2))))
        - k3 * Fw
    )

    # --- dy(11) cumulative biomass (MATLAB:524) --------------------------
    # NB: this is *value*, not derivative — it integrates total biomass over time.
    dy[10] = y[11] + y[12] + y[13] + y[14]

    # --- dy(12..15) biomass regions, dy(16..27) vacuole states ----------
    dy[11] = da_0_dt
    dy[12] = da_1_dt
    dy[13] = da_3_dt
    dy[14] = da_4_dt
    dy[15] = dn0_dt
    dy[16:25] = dn_interior   # n_1..n_9 in one shot
    dy[25] = dn_m_dt
    dy[26] = dphi_0_dt

    # --- dy(28) CO2 off-gas (MATLAB:550-552) -----------------------------
    total_X_CO2 = y[11] + y[12]
    CER = total_X_CO2 * q_co2 * V
    dy[27] = (
        ((60.0 * Fg * 44.0 * 1000.0) / 22.4) * C_CO2_in
        + CER
        - ((60.0 * Fg * 44.0 * 1000.0) / 22.4) * y[27]
    ) / (Vg * 28.97 * 1000.0 / 22.4)

    # --- dy(29) dissolved CO2 (MATLAB:557-560) ---------------------------
    Henrys_c_co2 = np.exp(11.25 - 395.9 / (y[7] - 175.9)) / (44.0 * 100.0)
    C_star_CO2 = (total_pressure * y[27]) / Henrys_c_co2
    dy[28] = kla * delta_c_0 * (C_star_CO2 - y[28]) - y[28] * dilution / y[4]

    # --- dy(30) PAA (MATLAB:564) -----------------------------------------
    dy[29] = (
        Fpaa * PAA_c / V
        - (Y_PAA_P * dP_dt)
        - Y_PAA_X * X_1
        - m_PAA * y[3]
        - y[29] * dilution / y[4]
    )

    # --- dy(31) NH3 / nitrogen (MATLAB:566-568) --------------------------
    X_C_nitrogen = (-r_b0 - r_e1 - r_d1 - r_d4) * Y_NX
    P_C_nitrogen = -dP_dt * Y_NP
    dy[30] = (
        (NH3_shots * N_conc_shot) / y[4]
        + X_C_nitrogen + P_C_nitrogen
        - m_N * total_X
        + (1.0 * N_conc_paa * Fpaa / y[4])
        + N_conc_oil * Foil / y[4]
        - y[30] * dilution / y[4]
    )

    # --- dy(32), dy(33) integrals of growth-rate parameters ------------
    # MATLAB allocates dy=zeros(31,1) at line 445 then silently grows when
    # writing dy(32), dy(33). We allocated 33 up-front, so just assign.
    dy[31] = mu_p
    dy[32] = mu_e

    return dy
