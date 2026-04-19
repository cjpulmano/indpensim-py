# How to capture the MATLAB golden reference (Phase 1)

This is the **only** thing you need to do in MATLAB during the entire port. After this, all work is Python-side.

## What you'll produce

For each reference batch you run, three files land under `data/matlab_reference/`:

- `batch_seed<S>_b<NN>_states.csv` — full per-sample trajectory: time + 33 ODE states + 12 manipulated inputs. Two header rows (field name, units), then ~570 sample rows.
- `batch_seed<S>_b<NN>_raman.csv` — 2200 Raman wavenumbers × N samples (only if Raman recording was enabled for that batch).
- `batch_seed<S>_b<NN>_meta.json` — seed, batch flags, sample count, MATLAB version. Used by Python validation harness to know what to load.

These are the immutable reference. The Python port is "done" when its trajectories match these CSVs within trajectory tolerances.

## Steps

### One-time setup
1. Install MATLAB (R2019a or later — same era as the original code). Required toolboxes: **Signal Processing** (`sgolayfilt`) and **Curve Fitting** (`smooth`). Nothing else.
2. Open MATLAB in this project root: `<repo-root>/IndPenSim_V2.02`.
3. Add the project to the path:
   ```matlab
   addpath(genpath(pwd))
   ```

### Per-batch capture
Run this in the MATLAB Command Window:

```matlab
rng(42)                              % SET THE SEED — identical Python runs need this
Generate_Production_Batch_data_V4    % runs the simulator, populates Raw_Batch_data in workspace
```

This takes a few minutes per batch. The default driver runs **2 batches** with these flags:

| Batch | Fault | Control          | Length    | Raman                    |
|-------|-------|------------------|-----------|--------------------------|
| 1     | none  | recipe-driven    | uneven    | record only              |
| 2     | yes   | operator-driven  | fixed     | use Raman to control PAA |

When it finishes (you'll see plot windows), dump the reference:

```matlab
seed_label  = 42;        % must match the rng() above
batch_index = 1;         % start with the no-fault recipe-driven batch
matlab_dump_reference    % from scripts/matlab_dump_reference.m
```

Then repeat with `batch_index = 2` to capture the fault batch.

### What seeds to capture

For the port, capture **at minimum**:
- `rng(42), batch 1` — clean baseline, recipe-driven, no fault. This is the primary validation target.
- `rng(42), batch 2` — fault + operator control. Tests fault-injection and controller paths.
- `rng(7),  batch 1` — second seed to confirm Python isn't accidentally fitted to seed 42's trajectory.

That's three reference CSVs ≈ 6 batch runs ≈ ~30–60 min of compute. Plenty of headroom on a 30-day trial.

## Sanity check

After dumping, verify the CSV looks right:

```matlab
T = readtable('data/matlab_reference/batch_seed42_b01_states.csv');
disp(T.Properties.VariableNames)        % should include S, DO2, ..., mu_X_calc, Fg, ..., Fremoved
disp(size(T))                           % (~571, 46)  — 1 time col + 33 states + 12 inputs
plot(T.time_h, T.P)                     % penicillin titre should rise to ~30-40 g/L over ~230h
```

If P is flat or NaN, something went wrong before the dump — not in the dump itself.

## Don't burn trial time on these mistakes

- **Forgot `rng()`.** If you didn't seed first, the trajectory is unreproducible and useless as a reference. Restart MATLAB and rerun.
- **Edited `indpensim_ode.m` for "debugging".** Don't. The reference must come from the unmodified original. If you suspect an issue, capture *both* the modified and original versions.
- **Closed MATLAB before dumping.** Workspace is gone. Rerun `Generate_Production_Batch_data_V4`.

## What to do once captured

Tell me — I'll run the Python validation harness against the CSVs.
