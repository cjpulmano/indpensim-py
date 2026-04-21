# indpensim-py

Python port of **IndPenSim V2.02** — an industrial-scale fed-batch penicillin
fermentation simulator. The original is MATLAB; this is a faithful end-to-end
Python port validated against MATLAB reference trajectories — mean
peak-normalized error under 1% on most states across 12 distinct configs
(see [VALIDATION.md](VALIDATION.md) for the full picture, including the
three channels where closed-loop solver amplification leaves wider bounds).

## What this is

- 33-state stiff ODE model of an industrial penicillin fermentation
- pH and temperature PID control, sequential-batch recipe driver,
  fault injection, PRBS noise
- Simulated Raman spectroscopy + PLS-based PAA concentration prediction
- Multi-batch campaign driver with CLI and CSV outputs

## Upstream / original

The original MATLAB simulator (which this port reproduces) is by Stephen Goldrick et al.

- Download:  http://www.industrialpenicillinsimulation.com/
- Paper:     Goldrick et al., "Modern day control challenges for industrial-scale
             fermentation processes," *Computers & Chemical Engineering*, 2019.
             https://doi.org/10.1016/j.compchemeng.2019.05.037
- Earlier:   Goldrick et al., "The Development of an Industrial Scale Fed-Batch
             Fermentation Simulation," *Journal of Biotechnology*, 2015.
             https://doi.org/10.1016/j.jbiotec.2014.10.029

## Install

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
```

(Or use `pip install -e .` if you don't have uv.)

## Run

Generate a campaign of batches and dump per-batch CSVs:

```bash
python -m indpensim.driver --num-batches 5 --seed 42 --out runs/
```

Or replay a captured MATLAB initial condition (for validation):

```bash
python -m indpensim.driver --from-capture --capture-seeds 42 --capture-batches 1 --out runs/
```

From Python:

```python
from indpensim.driver import batch_spec_from_python_rng, CampaignConfig, BatchConfig
from indpensim.simulation import simulate
import numpy as np

rng = np.random.default_rng(42)
spec = batch_spec_from_python_rng(rng, batch_no=1,
                                   campaign=CampaignConfig(),
                                   batch=BatchConfig(raman_spec=1))
result = simulate(spec)
print(result.history.channels["P"][-1])     # final penicillin g/L
```

## Layout

```
indpensim/
  driver.py           - multi-batch campaign + CLI
  simulation.py       - main loop (port of indpensim.m)
  ode/rhs.py          - 33-state ODE right-hand side (port of indpensim_ode.m)
  control/
    controller.py     - port of fctrl_indpensim.m (PID + SBC + faults + Raman PAA loop)
    pid.py            - port of PIDSimple3.m
    history.py        - per-channel batch trajectory container
  pat/
    raman.py          - simulated Raman spectrum (port of Raman_Sim.m)
    substrate.py      - PLS-based PAA prediction (port of Substrate_prediction.m)
    pls_model.py      - PAA_PLS_model.mat loader
  io/
    parameters.py     - 105-element parameter vector
    initial_conditions.py  - loader for MATLAB-captured initial conditions
  validation/
    playback.py       - replay a captured batch with MATLAB inputs
docs/
  state_vector.md     - 33-state glossary (Y(N) MATLAB <-> y[N-1] Python)
  parameters.md       - 105-parameter catalog
  pls_model.md        - PLS coefficient interpretation
  matlab_reference_capture.md  - how to capture MATLAB reference data
scripts/
  matlab_*.m          - MATLAB capture scripts
tests/                - 136 tests (states, controller, ODE, PLS, end-to-end)
```

## Validation

Tests compare against MATLAB-captured trajectories from the original simulator.
Three layers: ODE-only playback, single-seed end-to-end, and a 12-config
multi-seed suite covering each fault branch, the Raman closed-loop PAA
controller, and variable-length batches.

End-to-end batch trajectory matches MATLAB on the validated configs within
tight per-channel bounds (mean peak-normalized error <1% on most channels).
See [VALIDATION.md](VALIDATION.md) for methodology, the solver-tolerance
decision (production uses `rtol=1e-3` for speed; validation uses `rtol=1e-6`),
per-channel thresholds, and known limits.

Run the full suite:

```bash
pytest
```

## Faithful port notes

Three published-source quirks in the MATLAB are reproduced verbatim, not silently
fixed:

- `fctrl_indpensim.m:45` — `ph_err1` is missing the `pH_sp -` term (typo)
- `fctrl_indpensim.m:147` — heating-branch PID for Fh uses Fc history as `u_prev`
- `fctrl_indpensim.m:148-149` — dead write of `Fc=0` immediately overwritten

See the docstring in `indpensim/control/controller.py` for context.
