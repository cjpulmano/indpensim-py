"""Tests for parameter dataclasses + legacy par-vector conversion."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from indpensim.io.parameters import (
    KineticParameters,
    Parameters,
    ProcessParameters,
)


# Reference values pulled directly from Parameter_list.m for round-trip checks.
# par(1..105), 1-based MATLAB indices. Five entries are runtime-supplied; we
# pick concrete inputs so the result is a fully populated 105-element array.
RUNTIME = dict(mu_p=0.041, mux_max=0.41, alpha_kla=85.0, N_conc_paa=130000.0, PAA_c=530.0)


def test_default_factory_populates_all_groups():
    p = Parameters.default(**RUNTIME)
    # Each group must be present and immutable.
    for group_name in ("kinetics", "process", "thermal", "ph_ion", "nitrogen",
                       "paa", "k4", "co2_visc", "density"):
        assert getattr(p, group_name) is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.kinetics.mu_p = 99.0


def test_to_legacy_par_vector_length_and_runtime_slots():
    p = Parameters.default(**RUNTIME)
    v = p.to_legacy_par_vector()
    assert v.shape == (105,)
    assert v.dtype == np.float64
    # Spot-check the five runtime-supplied par() slots (1-based → 0-based).
    assert v[0]  == RUNTIME["mu_p"]          # par(1)
    assert v[1]  == RUNTIME["mux_max"]       # par(2)
    assert v[30] == RUNTIME["alpha_kla"]     # par(31)
    assert v[68] == RUNTIME["N_conc_paa"]    # par(69)
    assert v[74] == RUNTIME["PAA_c"]         # par(75)


def test_to_legacy_par_vector_known_constants():
    """Spot-check ~12 hard-coded values straight from Parameter_list.m."""
    p = Parameters.default(**RUNTIME)
    v = p.to_legacy_par_vector()

    expected = {
        # 1-based MATLAB index : exact value from Parameter_list.m
        3:   0.4,           # ratio_mu_e_mu_b
        20:  0.003,         # mu_h
        23:  1.85,          # Y_sX (was Y_sx in source)
        33:  0.34,          # b (kla exponent → kla_b)
        43:  8.314,         # R
        53:  4.18,          # C_pw
        66:  1e-5,          # K1
        77:  45.0,          # Y_PAA_X (= 37.5 * 1.2)
        81:  -64.29,        # B_1
        86:  0.89,          # delta_c_0 (was delta_c_o)
        87:  0.005,         # k3 (was k_3)
        92:  0.1353,        # q_co2 (= 0.123 * 1.1)
        96:  1540.0,        # rho_g (was pho_g)
        102: 0.033,         # C_CO2_in
        105: 2451.8,        # alpha_1
    }
    for matlab_idx, want in expected.items():
        got = v[matlab_idx - 1]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), (
            f"par({matlab_idx}) expected {want}, got {got}"
        )


def test_param_groups_are_frozen():
    k = KineticParameters(mu_p=0.04, mux_max=0.4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.mu_p = 0.05


def test_process_alpha_kla_required():
    """alpha_kla has no default — should raise without it."""
    with pytest.raises(TypeError):
        ProcessParameters()  # type: ignore[call-arg]
