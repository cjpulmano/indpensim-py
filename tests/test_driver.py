"""Driver tests — three shallow checks per the design doc.

(a) batch_spec_from_python_rng produces init values within expected
    distributions (statistical sanity, not bit-match);
(b) batch_spec_from_capture round-trips an existing capture;
(c) running 2 batches serially writes 2 CSVs with the right schema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
INITCONDS_MAT = REF_DIR / "batch_seed42_b01_initconds.mat"


def test_python_rng_produces_init_within_expected_distributions():
    """Draw 200 batches; check sample means/stds align with documented MATLAB
    distributions (indpensim_run.m:79-136). Wide tolerances — purely a sanity
    check that we wired distributions correctly."""
    from indpensim.driver import (
        BatchConfig, CampaignConfig, batch_spec_from_python_rng,
    )
    rng = np.random.default_rng(0)
    cfg = CampaignConfig()
    bcfg = BatchConfig()
    specs = [batch_spec_from_python_rng(rng, b, cfg, bcfg) for b in range(1, 201)]
    Vs   = np.array([s.initial_conditions.V   for s in specs])
    pHs  = np.array([s.initial_conditions.pH  for s in specs])
    NH3s = np.array([s.initial_conditions.NH3 for s in specs])
    PAAcs = np.array([s.initial_conditions.PAA_c for s in specs])

    # mean within ~3 SE; std within 30%.
    assert abs(Vs.mean()   - 5.800e4) < 3 * 500   / np.sqrt(200) * 5
    assert abs(pHs.mean()  - 6.5)     < 3 * 0.1   / np.sqrt(200) * 5
    assert 0.7 * 500 < Vs.std()  < 1.3 * 500
    assert 0.7 * 0.1 < pHs.std() < 1.3 * 0.1
    assert abs(NH3s.mean() - 1700)    < 3 * 50 / np.sqrt(200) * 5
    assert abs(PAAcs.mean() - 530000) < 3 * 20000 / np.sqrt(200) * 5


@pytest.mark.skipif(not INITCONDS_MAT.exists(),
                    reason="captured initconds .mat missing")
def test_capture_roundtrip():
    """batch_spec_from_capture must round-trip the captured init-condition file."""
    from indpensim.driver import batch_spec_from_capture
    spec = batch_spec_from_capture(seed=42, batch_no=1)
    assert spec.h == 0.2
    assert spec.T > 0
    assert spec.initial_conditions.V > 5e4
    assert spec.disturbances.distMuP.shape[0] == int(spec.T / spec.h) + 1
    assert spec.control_flags.Inhib in (0, 1, 2)


def test_run_campaign_writes_csv_with_correct_schema(tmp_path: Path):
    """Run two short batches and verify CSV columns + rows."""
    from indpensim.driver import (
        BatchConfig, CampaignConfig, batch_spec_from_python_rng, run_campaign,
    )
    # Use a SHORT batch length to keep the test fast — not the production default.
    cfg = CampaignConfig(optimum_T=20)
    bcfg = BatchConfig(raman_spec=0)
    rng = np.random.default_rng(123)
    specs = [batch_spec_from_python_rng(rng, b, cfg, bcfg) for b in (1, 2)]
    result = run_campaign(specs, tmp_path)

    assert len(result.batch_csvs) == 2
    for path in result.batch_csvs:
        assert path.exists()
        df = pd.read_csv(path, header=[0, 1])
        df.columns = df.columns.get_level_values(0)
        # 1 (time) + 33 (states) + 12 (controls) = 46 columns
        assert df.shape[1] == 46
        # 100 samples for a 20h batch at h=0.2
        assert df.shape[0] == 100
        # Spot-check a few required columns
        for required in ("time_h", "S", "DO2", "pH", "Fc", "Fb", "mu_X_calc"):
            assert required in df.columns
    # Summary
    summary_csv = tmp_path / "campaign_summary.csv"
    assert summary_csv.exists()
    summary = pd.read_csv(summary_csv)
    assert list(summary["batch_no"]) == [1, 2]
    assert "P_final_g_per_L" in summary.columns
