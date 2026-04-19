"""Multi-batch campaign driver — port of Generate_Production_Batch_data_V4.m.

Two entry points produce the same ``CapturedBatch`` shape that ``simulate()``
consumes:

  - ``batch_spec_from_capture(seed, batch_no)`` — loads a MATLAB-captured
    init-condition .mat. Use this for validation against MATLAB.
  - ``batch_spec_from_python_rng(rng, batch_no, campaign, batch)`` — generates
    a fresh init using numpy's RNG. Use this for production runs that don't
    need MATLAB equivalence.

The Python-RNG path **intentionally diverges** from MATLAB's per-call
``rng(Seed_ref + Batch_no + Rand_ref)`` pattern. Instead it uses one
``np.random.default_rng(seed + batch_no)`` and draws sequentially. This
costs nothing (we already lose bit-equivalence to MATLAB on this path)
and avoids the fragility of order-dependent re-seeding.

CLI:
    python -m indpensim.driver --num-batches 5 --seed 42 --out runs/
    python -m indpensim.driver --from-capture --seeds 42 --batches 1 --out runs/
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from indpensim.io.initial_conditions import (
    CapturedBatch, ControlFlags, DisturbanceTrajectories, InitialConditions,
    load_captured_batch,
)
from indpensim.simulation import SimulationResult, simulate


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CampaignConfig:
    """Campaign-level constants — same for every batch in a run.

    Defaults match Generate_Production_Batch_data_V4.m + indpensim_run.m.
    """
    h: float = 0.2                         # sample period [h]
    optimum_T: int = 230                   # nominal batch length [h]
    batch_length_sd: float = 25.0          # std of length variation when fixed_length=False
    T_setpoint: float = 298.0              # temperature SP [K]
    pH_setpoint: float = 6.5               # pH SP
    Off_line_m: int = 12                   # off-line meas sample period [h]
    Off_line_delay: int = 4                # off-line meas analysis delay [h]
    inhib: int = 2                         # inhibition mode (0/1/2)
    dis: int = 1                           # disturbance flag (0/1)


@dataclass(frozen=True)
class BatchConfig:
    """Per-batch overrides selecting fault modes, control strategy, etc."""
    faults: int = 0                        # 0..8 — see fctrl_indpensim.m
    prbs: int = 0                          # 0=SBC recipe, 1=PRBS noise
    fixed_length: bool = True              # False → length varies by 25*randn
    raman_spec: int = 0                    # 0=none, 1=record, 2=close PAA loop


# ---------------------------------------------------------------------------
# Init-condition generators
# ---------------------------------------------------------------------------

def _draw_initial_conditions(rng: np.random.Generator) -> InitialConditions:
    """Mirror indpensim_run.m:79-136. Distributions match exactly; the per-call
    seeding pattern does NOT (we use one rng, drawn sequentially)."""
    intial_conds = 0.5 + 0.05 * rng.standard_normal()
    mux = 0.41 + 0.025 * rng.standard_normal()
    mup = 0.041 + 0.0025 * rng.standard_normal()
    return InitialConditions(
        S=1.0 + 0.1 * rng.standard_normal(),
        DO2=15.0 + 0.5 * rng.standard_normal(),
        X=intial_conds + 0.1 * rng.standard_normal(),
        P=0.0,
        V=5.800e4 + 500 * rng.standard_normal(),
        Wt=6.2e4 + 500 * rng.standard_normal(),
        CO2outgas=0.038 + 0.001 * rng.standard_normal(),
        O2=0.20 + 0.05 * rng.standard_normal(),
        pH=6.5 + 0.1 * rng.standard_normal(),
        T=297.0 + 0.5 * rng.standard_normal(),
        a0=intial_conds * (1 / 3),
        a1=intial_conds * (2 / 3),
        a3=0.0, a4=0.0, Culture_age=0.0,
        PAA=1400.0 + 50 * rng.standard_normal(),
        NH3=1700.0 + 50 * rng.standard_normal(),
        mup=mup, mux=mux,
        alpha_kla=85.0 + 10 * rng.standard_normal(),
        PAA_c=530000.0 + 20000 * rng.standard_normal(),
        N_conc_paa=2 * 75000.0 + 2000 * rng.standard_normal(),
    )


def _draw_disturbance_trajectories(
    rng: np.random.Generator, N_plus_1: int,
) -> DisturbanceTrajectories:
    """Low-pass filtered Gaussian noise — mirror indpensim_run.m:142-183.

    MATLAB:
        b1 = 1 - 0.995; a1 = [1, -0.995];
        v = randn(T/h+1, 1); distMuP = filter(b1, a1, 0.03*v);
    scipy.signal.lfilter uses the same convention as MATLAB's `filter`.
    """
    b = np.array([1 - 0.995])
    a = np.array([1.0, -0.995])

    def _draw(scale: float) -> np.ndarray:
        return lfilter(b, a, scale * rng.standard_normal(N_plus_1))

    return DisturbanceTrajectories(
        distMuP=_draw(0.03),
        distMuX=_draw(0.25),
        distcs=_draw(5 * 300),
        distcoil=_draw(300),
        distabc=_draw(0.2),
        distPAA=_draw(300000),
        distTcin=_draw(100),
        distO_2in=_draw(0.02),
    )


def _build_control_flags(
    batch_no: int, campaign: CampaignConfig, batch: BatchConfig,
) -> ControlFlags:
    return ControlFlags(
        SBC=0, PRBS=batch.prbs,
        Fixed_Batch_length=int(batch.fixed_length),
        IC=0,
        Inhib=campaign.inhib, Dis=campaign.dis,
        Faults=batch.faults, Vis=0,
        Raman_spec=batch.raman_spec,
        Batch_Num=batch_no,
        T_sp=campaign.T_setpoint, pH_sp=campaign.pH_setpoint,
        Off_line_m=campaign.Off_line_m, Off_line_delay=campaign.Off_line_delay,
    )


def batch_spec_from_python_rng(
    rng: np.random.Generator,
    batch_no: int,
    campaign: CampaignConfig,
    batch: BatchConfig,
) -> CapturedBatch:
    """Generate one batch spec using numpy RNG."""
    if batch.fixed_length:
        T = campaign.optimum_T
    else:
        T = int(round(campaign.optimum_T + campaign.batch_length_sd * rng.standard_normal()))

    ic = _draw_initial_conditions(rng)
    N_plus_1 = int(round(T / campaign.h)) + 1
    dist = _draw_disturbance_trajectories(rng, N_plus_1)
    flags = _build_control_flags(batch_no, campaign, batch)
    return CapturedBatch(
        initial_conditions=ic, disturbances=dist, control_flags=flags,
        h=campaign.h, T=T, Random_seed_ref=0, Seed_ref=0,
    )


def batch_spec_from_capture(seed: int, batch_no: int) -> CapturedBatch:
    """Load a MATLAB-captured init for validation runs."""
    return load_captured_batch(seed=seed, batch_index=batch_no)


# ---------------------------------------------------------------------------
# Output writer (matches matlab_dump_reference.m schema)
# ---------------------------------------------------------------------------

# Order matches matlab_dump_reference.m:42-51 exactly.
_STATE_FIELDS = (
    "S", "DO2", "O2", "P", "V", "Wt", "pH", "T", "Q", "Viscosity",
    "Culture_age", "a0", "a1", "a3", "a4",
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9",
    "nm", "phi0", "CO2outgas", "CO2_d", "PAA", "NH3",
    "mu_P_calc", "mu_X_calc",
)
_INPUT_FIELDS = (
    "Fg", "RPM", "Fpaa", "Fs", "Fa", "Fb", "Fc", "Foil", "Fh", "Fw",
    "pressure", "Fremoved",
)
# Units row 2 — matches matlab_dump_reference.m's units. Most are unset there,
# so we use empty strings except for the ones we know.
_UNITS = {
    "time_h": "h", "S": "g/L", "DO2": "mg/L", "O2": "%", "P": "g/L",
    "V": "L", "Wt": "kg", "pH": "-", "T": "K", "Q": "kJ", "Viscosity": "cP",
    "Culture_age": "h", "PAA": "mg/L", "NH3": "mg/L",
}


def write_batch_csv(result: SimulationResult, out_path: Path) -> None:
    """Write one batch trajectory in matlab_dump_reference schema (2-row header)."""
    h = result.history
    N = h.N
    columns = ["time_h", *_STATE_FIELDS, *_INPUT_FIELDS]
    units_row = [_UNITS.get(c, "") for c in columns]

    data = {"time_h": result.t[1 : N + 1]}
    for f in _STATE_FIELDS:
        data[f] = h.channels[f][1 : N + 1]
    for f in _INPUT_FIELDS:
        data[f] = h.channels[f][1 : N + 1]

    df = pd.DataFrame(data, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        fh.write(",".join(columns) + "\n")
        fh.write(",".join(units_row) + "\n")
        df.to_csv(fh, header=False, index=False)


# ---------------------------------------------------------------------------
# Campaign runner
# ---------------------------------------------------------------------------

@dataclass
class CampaignResult:
    out_dir: Path
    batch_csvs: list[Path]
    summary: pd.DataFrame                  # one row per batch


def _summary_row(batch_no: int, result: SimulationResult) -> dict:
    """Penicillin yield + batch length, matching MATLAB Summmary_of_campaign."""
    h = result.history
    N = h.N
    P = h.channels["P"][1 : N + 1]
    V = h.channels["V"][1 : N + 1]
    Fremoved = h.channels["Fremoved"][1 : N + 1]
    P_end = float(P[-1])
    # Crude integral of penicillin removed during batch (Fremoved is negative when discharging).
    P_harvested = float(np.sum(np.maximum(-Fremoved, 0) * P) * (result.t[1] - result.t[0]))
    return {
        "batch_no": batch_no,
        "P_final_g_per_L": P_end,
        "P_harvested_g": P_harvested,
        "V_final_L": float(V[-1]),
        "duration_h": float(result.t[N]),
    }


def run_campaign(
    specs: list[CapturedBatch],
    out_dir: Path,
) -> CampaignResult:
    """Run a sequence of batches and write per-batch CSVs serially."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csvs: list[Path] = []
    summary_rows: list[dict] = []
    for i, spec in enumerate(specs, start=1):
        result = simulate(spec)
        csv_path = out_dir / f"batch_{i:02d}.csv"
        write_batch_csv(result, csv_path)
        csvs.append(csv_path)
        summary_rows.append(_summary_row(i, result))
        print(f"  batch {i:02d}: T={spec.T}h  P_final={summary_rows[-1]['P_final_g_per_L']:.3f} g/L")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "campaign_summary.csv", index=False)
    return CampaignResult(out_dir=out_dir, batch_csvs=csvs, summary=summary)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="IndPenSim multi-batch driver")
    p.add_argument("--num-batches", type=int, default=2,
                   help="number of batches to generate (Python-RNG mode)")
    p.add_argument("--seed", type=int, default=42,
                   help="master RNG seed (Python-RNG mode)")
    p.add_argument("--from-capture", action="store_true",
                   help="load init from data/matlab_reference instead of generating")
    p.add_argument("--capture-seeds", type=int, nargs="+", default=[42],
                   help="when --from-capture, the seed labels to load")
    p.add_argument("--capture-batches", type=int, nargs="+", default=[1],
                   help="when --from-capture, the batch indices to load")
    p.add_argument("--out", type=Path, default=Path("runs"),
                   help="output directory for per-batch CSVs")
    p.add_argument("--raman-spec", type=int, choices=(0, 1, 2), default=0,
                   help="0=no Raman, 1=record only, 2=close PAA loop (Python-RNG mode)")
    p.add_argument("--faults", type=int, choices=range(0, 9), default=0,
                   metavar="0..8", help="fault mode (Python-RNG mode)")
    p.add_argument("--variable-length", action="store_true",
                   help="randomize batch length around T=230±25 (Python-RNG mode)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.from_capture:
        specs = []
        for s in args.capture_seeds:
            for b in args.capture_batches:
                specs.append(batch_spec_from_capture(s, b))
        print(f"loaded {len(specs)} captured batch(es)")
    else:
        rng = np.random.default_rng(args.seed)
        campaign = CampaignConfig()
        batch_cfg = BatchConfig(
            faults=args.faults, prbs=0,
            fixed_length=not args.variable_length,
            raman_spec=args.raman_spec,
        )
        specs = [
            batch_spec_from_python_rng(rng, batch_no, campaign, batch_cfg)
            for batch_no in range(1, args.num_batches + 1)
        ]
        print(f"generated {len(specs)} batch spec(s) from seed={args.seed}")

    result = run_campaign(specs, args.out)
    print(f"\nwrote {len(result.batch_csvs)} CSV(s) + campaign_summary.csv to {result.out_dir}")
    print(result.summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
