"""Quick sanity check on the MATLAB reference CSVs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path("data/matlab_reference")


def main() -> None:
    states = pd.read_csv(REF / "batch_seed42_b01_states.csv", header=[0, 1])
    states.columns = states.columns.get_level_values(0)
    print(f"states: {states.shape[0]} rows × {states.shape[1]} cols")
    print(f"  time range: {states['time_h'].iloc[0]:.2f}h .. "
          f"{states['time_h'].iloc[-1]:.2f}h "
          f"(Δt = {states['time_h'].iloc[1] - states['time_h'].iloc[0]:.3g}h)")
    print()
    print("  key trajectories (start → end):")
    for col in ["S", "DO2", "P", "V", "pH", "T",
                "a0", "a1", "a3", "a4", "PAA", "NH3", "Viscosity"]:
        v0, vN = states[col].iloc[0], states[col].iloc[-1]
        vmin, vmax = states[col].min(), states[col].max()
        print(f"    {col:>12}: {v0:>12.4g}  →  {vN:>12.4g}   "
              f"(range {vmin:.3g} .. {vmax:.3g})")

    print()
    print("  pH semantics check:")
    pH = states["pH"].dropna()
    print(f"    raw pH column range: {pH.min():.4g} .. {pH.max():.4g}")
    if pH.min() > 1:
        print("    ⇒ stored as pH directly (range 4–8 typical)")
    else:
        print("    ⇒ stored as [H+] in mol/L (range 1e-8..1e-4 typical)")

    raman = pd.read_csv(REF / "batch_seed42_b01_raman.csv")
    print()
    print(f"raman: {raman.shape[0]} wavenumbers × {raman.shape[1]} cols "
          f"(1 wavelength + {raman.shape[1]-1} samples)")
    print(f"  wavelength range: {raman.iloc[0,0]:.0f} .. {raman.iloc[-1,0]:.0f} cm⁻¹")
    sample1 = raman.iloc[:, 1].to_numpy()
    print(f"  sample 1 intensity: min={sample1.min():.4g}  max={sample1.max():.4g}  "
          f"mean={sample1.mean():.4g}")

    meta = json.loads((REF / "batch_seed42_b01_meta.json").read_text())
    print()
    print(f"meta: {meta}")

    # Cross-check sample count alignment
    n_states = states.shape[0]
    n_raman = raman.shape[1] - 1
    if n_states != n_raman:
        print(f"\n⚠ states samples={n_states} but raman samples={n_raman} — mismatch!")
    else:
        print(f"\n✓ sample counts align: {n_states}")


if __name__ == "__main__":
    main()
