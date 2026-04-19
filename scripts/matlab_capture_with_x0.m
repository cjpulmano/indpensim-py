%% matlab_capture_with_x0.m
% One-shot: re-runs ONE batch with full instrumentation so we capture not
% just the trajectory but also the random initial conditions (x0), the
% randomized parameters (alpha_kla, PAA_c, N_conc_paa), and the per-sample
% disturbance trajectories (distMuP, distMuX, distcs, distcoil, distabc,
% distPAA, distTcin, distO_2in). All of these are needed to seed the Python
% port for bit-exact end-to-end trajectory matching.
%
% Without these, Python validation can only match the disturbance-insensitive
% subset of states (Wt, P, Viscosity, a3, a4, PAA, NH3) — that's enough to
% prove the ODE structure is correct, but not enough for a full regression
% test on every state.
%
% Usage:
%   1. addpath(genpath(pwd))
%   2. seed_label = 42;     % or whatever seed you want a reference for
%      batch_index = 1;
%      matlab_capture_with_x0
%   3. data/matlab_reference/batch_seed<N>_b<NN>_initconds.mat appears.
%      Contains: x0, alpha_kla, PAA_c, N_conc_paa, Random_seed_ref, Xinterp.
%   4. Then re-run scripts/matlab_dump_reference.m as before to refresh the
%      states/raman CSVs from this same workspace (so they're guaranteed
%      consistent with the saved init conditions).
%
% This script reproduces the indpensim_run.m setup logic INLINE so we can
% capture the locals before they're discarded. The original .m files are
% untouched.

if ~exist('seed_label', 'var');  error('Set seed_label before running.'); end
if ~exist('batch_index', 'var'); error('Set batch_index before running.'); end

clearvars -except seed_label batch_index
out_dir = fullfile(pwd, 'data', 'matlab_reference');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% Reproduce the master seed used by Generate_Production_Batch_data_V4
rng(seed_label)

% --- Mirror the relevant parts of Generate_Production_Batch_data_V4 ----
Operational_days = 336;
Bioreactor_turn_around_time = 3;
Batch_run_flags.Batch_fault_order_reference = [0,1];
Batch_run_flags.Control_strategy            = [0,1];
Batch_run_flags.Batch_length                = [1,0];
Batch_run_flags.Raman_spec                  = [1,2];

% --- Mirror indpensim_run.m setup so we can grab x0 BEFORE indpensim runs.
% (Copy of lines 19-141 of indpensim_run.m; original file is untouched.)
Batch_no = batch_index;
Ctrl_flags.SBC = 0;
Ctrl_flags.PRBS = Batch_run_flags.Control_strategy(Batch_no);
Ctrl_flags.Fixed_Batch_length = Batch_run_flags.Batch_length(Batch_no);
Ctrl_flags.IC = 0;
Ctrl_flags.Inhib = 2;
Ctrl_flags.Dis = 1;
Ctrl_flags.Faults = Batch_run_flags.Batch_fault_order_reference(Batch_no);
Ctrl_flags.Vis = 0;
Ctrl_flags.Raman_spec = Batch_run_flags.Raman_spec(Batch_no);
Ctrl_flags.Batch_Num = Batch_no;
Ctrl_flags.Off_line_m = 12;
Ctrl_flags.Off_line_delay = 4;
Ctrl_flags.plots = 0;     % disable plots during instrumented capture

Ctrl_flags.SBC = 0;
Ctrl_flags.Vis = 0;
Optimum_Batch_lenght = 230;
if Ctrl_flags.Fixed_Batch_length == 1
    Batch_length_variation = 25 * randn(1);
    T = round(Optimum_Batch_lenght + Batch_length_variation);
else
    T = Optimum_Batch_lenght;
end
Random_seed_ref = ceil(rand * 1000);
Seed_ref = 31 + Random_seed_ref;
Rand_ref = 1;

rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
intial_conds = 0.5 + 0.05 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.mux = 0.41 + 0.025 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.mup = 0.041 + 0.0025 * randn;
h = 0.2;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.S = 1 + 0.1 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.DO2 = 15 + 0.5 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.X = intial_conds + 0.1 * randn;
x0.P = 0;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.V = 5.800e+04 + 500 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.Wt = 6.2e+04 + 500 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.CO2outgas = 0.038 + 0.001 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.O2 = 0.20 + 0.05 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.pH = 6.5 + 0.1 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.T = 297 + 0.5 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.a0 = intial_conds * (1/3);
x0.a1 = intial_conds * (2/3);
x0.a3 = 0;
x0.a4 = 0;
x0.Culture_age = 0;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.PAA = 1400 + 50 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
x0.NH3 = 1700 + 50 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
alpha_kla = 85 + 10 * randn;
rng(Seed_ref + Batch_no + Rand_ref); Rand_ref = Rand_ref + 1;
PAA_c = 530000 + 20000 * randn;
rng(Seed_ref + Batch_no + Rand_ref);
N_conc_paa = 2 * 75000 + 2000 * randn;

Batch_time = 0:h:T;
Ctrl_flags.T_sp = 298;
Ctrl_flags.pH_sp = 6.5;

% Disturbance trajectories (lines 142-183 of indpensim_run.m)
rng(Random_seed_ref + Batch_no);
b1 = 1 - 0.995;
a1 = [1, -0.995];
v = randn(T/h + 1, 1); distMuP   = filter(b1, a1, 0.03   * v);
v = randn(T/h + 1, 1); distMuX   = filter(b1, a1, 0.25   * v);
v = randn(T/h + 1, 1); distcs    = filter(b1, a1, 5*300  * v);
v = randn(T/h + 1, 1); distcoil  = filter(b1, a1, 300    * v);
v = randn(T/h + 1, 1); distabc   = filter(b1, a1, 0.2    * v);
v = randn(T/h + 1, 1); distPAA   = filter(b1, a1, 300000 * v);
v = randn(T/h + 1, 1); distPAA   = filter(b1, a1, 300000 * v);     % source overwrites; we mirror
v = randn(T/h + 1, 1); distTcin  = filter(b1, a1, 100    * v);
v = randn(T/h + 1, 1); distO_2in = filter(b1, a1, 0.02   * v);

% Save EVERYTHING needed to seed the Python port
out_path = fullfile(out_dir, sprintf('batch_seed%d_b%02d_initconds.mat', ...
                                      seed_label, batch_index));
save(out_path, 'x0', 'alpha_kla', 'PAA_c', 'N_conc_paa', ...
               'Random_seed_ref', 'Seed_ref', 'h', 'T', ...
               'Batch_time', 'Ctrl_flags', ...
               'distMuP', 'distMuX', 'distcs', 'distcoil', ...
               'distabc', 'distPAA', 'distTcin', 'distO_2in');

fprintf('Wrote %s\n', out_path);
fprintf('  x0 fields: %s\n', strjoin(fieldnames(x0)', ', '));
fprintf('  x0.S=%.4f  x0.DO2=%.4f  x0.X=%.4f  x0.V=%.1f  x0.pH=%.4f  x0.T=%.4f\n', ...
        x0.S, x0.DO2, x0.X, x0.V, x0.pH, x0.T);
fprintf('  x0.mup=%.6f  x0.mux=%.6f\n', x0.mup, x0.mux);
fprintf('  alpha_kla=%.4f  PAA_c=%.1f  N_conc_paa=%.1f\n', alpha_kla, PAA_c, N_conc_paa);
fprintf('  T (batch length)=%d  Random_seed_ref=%d\n', T, Random_seed_ref);
fprintf('\n');
fprintf('Now run scripts/matlab_dump_reference.m if you want to refresh the\n');
fprintf('states/raman CSVs from a fresh end-to-end simulation.\n');
