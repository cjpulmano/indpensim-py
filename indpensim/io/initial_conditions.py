"""Loader for MATLAB-captured initial conditions and disturbance trajectories.

Loads the file written by `scripts/matlab_capture_with_x0.m`:
    data/matlab_reference/batch_seed<S>_b<NN>_initconds.mat

Provides everything the Python port needs to seed a simulation that
matches MATLAB bit-for-bit (modulo solver tolerance):
  - x0: initial state values
  - alpha_kla, PAA_c, N_conc_paa: randomized parameters
  - per-sample disturbance trajectories (8 of them)
  - control flags (Inhib, Dis, Vis, setpoints, etc.)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


@dataclass(frozen=True)
class InitialConditions:
    # Initial state values (from x0 struct)
    S: float
    DO2: float
    X: float          # initial *total* biomass — used to seed a0/a1/a3/a4
    P: float
    V: float
    Wt: float
    CO2outgas: float
    O2: float
    pH: float         # in pH units (gets converted to [H+] inside indpensim.m)
    T: float
    a0: float
    a1: float
    a3: float
    a4: float
    Culture_age: float
    PAA: float
    NH3: float
    mup: float        # initial penicillin growth rate parameter
    mux: float        # initial biomass growth rate parameter

    # Randomized model parameters (passed to Parameter_list)
    alpha_kla: float
    PAA_c: float
    N_conc_paa: float


@dataclass(frozen=True)
class DisturbanceTrajectories:
    """Per-sample disturbance trajectories (length = T/h + 1)."""
    distMuP: np.ndarray
    distMuX: np.ndarray
    distcs: np.ndarray
    distcoil: np.ndarray
    distabc: np.ndarray
    distPAA: np.ndarray
    distTcin: np.ndarray
    distO_2in: np.ndarray

    def at(self, k: int) -> np.ndarray:
        """Return the 8-element disturbance vector at sample k (0-based).

        Order matches the inp1 controller-input slots [17..24] expected by
        ``ode.rhs.rhs``.
        """
        return np.array([
            self.distMuP[k], self.distMuX[k], self.distcs[k],
            self.distcoil[k], self.distabc[k], self.distPAA[k],
            self.distTcin[k], self.distO_2in[k],
        ], dtype=float)


@dataclass(frozen=True)
class ControlFlags:
    SBC: int
    PRBS: int
    Fixed_Batch_length: int
    IC: int
    Inhib: int
    Dis: int
    Faults: int
    Vis: int
    Raman_spec: int
    Batch_Num: int
    T_sp: float
    pH_sp: float
    Off_line_m: int
    Off_line_delay: int


@dataclass(frozen=True)
class CapturedBatch:
    initial_conditions: InitialConditions
    disturbances: DisturbanceTrajectories
    control_flags: ControlFlags
    h: float           # sample period [h]
    T: int             # batch length [h]
    Random_seed_ref: int
    Seed_ref: int


_DEFAULT_REF_DIR = Path(__file__).resolve().parents[2] / "data" / "matlab_reference"


def load_captured_batch(seed: int, batch_index: int,
                         ref_dir: Path | str | None = None) -> CapturedBatch:
    """Load a `batch_seed<S>_b<NN>_initconds.mat` capture.

    Args:
        seed: matches the rng() call before generation (e.g. 42).
        batch_index: 1 or 2 in the default driver.
        ref_dir: directory containing the .mat. Default is data/matlab_reference.
    """
    base = Path(ref_dir) if ref_dir is not None else _DEFAULT_REF_DIR
    path = base / f"batch_seed{seed}_b{batch_index:02d}_initconds.mat"
    if not path.exists():
        raise FileNotFoundError(f"missing capture: {path}")
    raw = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)

    x0 = raw["x0"]
    cf = raw["Ctrl_flags"]

    ic = InitialConditions(
        S=float(x0.S), DO2=float(x0.DO2), X=float(x0.X), P=float(x0.P),
        V=float(x0.V), Wt=float(x0.Wt), CO2outgas=float(x0.CO2outgas),
        O2=float(x0.O2), pH=float(x0.pH), T=float(x0.T),
        a0=float(x0.a0), a1=float(x0.a1), a3=float(x0.a3), a4=float(x0.a4),
        Culture_age=float(x0.Culture_age), PAA=float(x0.PAA), NH3=float(x0.NH3),
        mup=float(x0.mup), mux=float(x0.mux),
        alpha_kla=float(raw["alpha_kla"]),
        PAA_c=float(raw["PAA_c"]),
        N_conc_paa=float(raw["N_conc_paa"]),
    )
    dist = DisturbanceTrajectories(
        distMuP=np.asarray(raw["distMuP"], dtype=float),
        distMuX=np.asarray(raw["distMuX"], dtype=float),
        distcs=np.asarray(raw["distcs"], dtype=float),
        distcoil=np.asarray(raw["distcoil"], dtype=float),
        distabc=np.asarray(raw["distabc"], dtype=float),
        distPAA=np.asarray(raw["distPAA"], dtype=float),
        distTcin=np.asarray(raw["distTcin"], dtype=float),
        distO_2in=np.asarray(raw["distO_2in"], dtype=float),
    )
    flags = ControlFlags(
        SBC=int(cf.SBC), PRBS=int(cf.PRBS),
        Fixed_Batch_length=int(cf.Fixed_Batch_length), IC=int(cf.IC),
        Inhib=int(cf.Inhib), Dis=int(cf.Dis), Faults=int(cf.Faults),
        Vis=int(cf.Vis), Raman_spec=int(cf.Raman_spec),
        Batch_Num=int(cf.Batch_Num),
        T_sp=float(cf.T_sp), pH_sp=float(cf.pH_sp),
        Off_line_m=int(cf.Off_line_m), Off_line_delay=int(cf.Off_line_delay),
    )
    return CapturedBatch(
        initial_conditions=ic,
        disturbances=dist,
        control_flags=flags,
        h=float(raw["h"]),
        T=int(raw["T"]),
        Random_seed_ref=int(raw["Random_seed_ref"]),
        Seed_ref=int(raw["Seed_ref"]),
    )
