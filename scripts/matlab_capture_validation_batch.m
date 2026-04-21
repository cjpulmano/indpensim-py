%% matlab_capture_validation_batch.m
% Single-batch capture + reference-trajectory dump, parametrized by config.
% Replaces the two-step (capture_with_x0 + dump_reference) dance for
% validation runs. One rng(seed) call, one indpensim() invocation, three
% output files — nothing duplicated, nothing to re-sync.
%
% Usage (set these workspace vars, then run this script):
%   seed_label              -- int, goes into rng(seed_label) and the filenames
%   faults_override         -- 0..8 (see Ctrl_flags.Faults)
%   raman_override          -- 0 | 1 | 2
%   fixed_length_override   -- 1 = fixed 230h, 0 = randomised
%
% Outputs (under data/matlab_reference/, using existing naming):
%   batch_seed<S>_b01_initconds.mat   -- x0, randomised params, disturbances, flags
%   batch_seed<S>_b01_states.csv      -- 33 states + 12 inputs, 1 row per sample
%   batch_seed<S>_b01_meta.json       -- seed, config, n_samples

if ~exist('seed_label','var');            error('Set seed_label before running.'); end
if ~exist('faults_override','var');       error('Set faults_override before running.'); end
if ~exist('raman_override','var');        error('Set raman_override before running.'); end
if ~exist('fixed_length_override','var'); error('Set fixed_length_override before running.'); end

% Resolve paths from THIS file's location so run()-induced cd's don't
% confuse us, and keep caller's workspace (configs, loop index) intact.
this_dir  = fileparts(mfilename('fullpath'));
repo_root = fileparts(this_dir);
out_dir   = fullfile(repo_root, 'data', 'matlab_reference');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

Batch_no = 1;

rng(seed_label)

% ---- Setup block (inlined from indpensim_run.m:20-141) -------------------
Ctrl_flags.SBC                = 0;
Ctrl_flags.PRBS               = 0;                        % recipe-driven
Ctrl_flags.Fixed_Batch_length = fixed_length_override;
Ctrl_flags.IC                 = 0;
Ctrl_flags.Inhib              = 2;
Ctrl_flags.Dis                = 1;
Ctrl_flags.Faults             = faults_override;
Ctrl_flags.Vis                = 0;
Ctrl_flags.Raman_spec         = raman_override;
Ctrl_flags.Batch_Num          = Batch_no;
Ctrl_flags.Off_line_m         = 12;
Ctrl_flags.Off_line_delay     = 4;
Ctrl_flags.plots              = 0;

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

% ---- Disturbance trajectories (inlined from indpensim_run.m:142-183) -----
rng(Random_seed_ref + Batch_no);
b1 = 1 - 0.995;
a1 = [1, -0.995];
v = randn(T/h + 1, 1); distMuP   = filter(b1, a1, 0.03   * v);
Xinterp.distMuP   = createChannel('Penicillin specific growth rate disturbance','g/Lh','h',Batch_time,distMuP);
v = randn(T/h + 1, 1); distMuX   = filter(b1, a1, 0.25   * v);
Xinterp.distMuX   = createChannel('Biomass specific growth rate disturbance','hr^{-1}','h',Batch_time,distMuX);
v = randn(T/h + 1, 1); distcs    = filter(b1, a1, 5*300  * v);
Xinterp.distcs    = createChannel('Substrate concentration disturbance',' g L^{-1}','h',Batch_time,distcs);
v = randn(T/h + 1, 1); distcoil  = filter(b1, a1, 300    * v);
Xinterp.distcoil  = createChannel('Oil concentration disturbance',' g L^{-1}','h',Batch_time,distcoil);
v = randn(T/h + 1, 1); distabc   = filter(b1, a1, 0.2    * v);
Xinterp.distabc   = createChannel('Acid/Base concentration disturbance','mol L^{-1}','h',Batch_time,distabc);
v = randn(T/h + 1, 1); distPAA   = filter(b1, a1, 300000 * v);
Xinterp.distPAA   = createChannel('PAA concentration disturbance',' g L^{-1}','h',Batch_time,distPAA);
v = randn(T/h + 1, 1); distPAA   = filter(b1, a1, 300000 * v);  % source overwrites; we mirror
Xinterp.distPAA   = createChannel('PAA concentration disturbance',' g L^{-1}','h',Batch_time,distPAA);
v = randn(T/h + 1, 1); distTcin  = filter(b1, a1, 100    * v);
Xinterp.distTcin  = createChannel('Coolant inlet temperature disturbance','K','h',Batch_time,distTcin);
v = randn(T/h + 1, 1); distO_2in = filter(b1, a1, 0.02   * v);
Xinterp.distO_2in = createChannel('Oxygen inlet concentration','%','h',Batch_time,distO_2in);

% ---- Parameters + simulation --------------------------------------------
par = Parameter_list(x0, alpha_kla, N_conc_paa, PAA_c);
fprintf('seed=%d faults=%d raman=%d fixed=%d  running indpensim (T=%dh)...\n', ...
        seed_label, faults_override, raman_override, fixed_length_override, T);
Xref = indpensim(@fctrl_indpensim, Xinterp, x0, h, T, 2, par, Ctrl_flags);

% ---- Save initconds (same schema as matlab_capture_with_x0.m) -----------
ic_path = fullfile(out_dir, sprintf('batch_seed%d_b01_initconds.mat', seed_label));
save(ic_path, 'x0', 'alpha_kla', 'PAA_c', 'N_conc_paa', ...
              'Random_seed_ref', 'Seed_ref', 'h', 'T', ...
              'Batch_time', 'Ctrl_flags', ...
              'distMuP', 'distMuX', 'distcs', 'distcoil', ...
              'distabc', 'distPAA', 'distTcin', 'distO_2in');
fprintf('  wrote %s\n', ic_path);

% ---- Dump states/inputs CSV (same schema as matlab_dump_reference.m) -----
state_fields = { ...
    'S','DO2','O2','P','V','Wt','pH','T','Q','Viscosity', ...
    'Culture_age','a0','a1','a3','a4', ...
    'n0','n1','n2','n3','n4','n5','n6','n7','n8','n9', ...
    'nm','phi0','CO2outgas','CO2_d','PAA','NH3', ...
    'mu_P_calc','mu_X_calc'};
input_fields = { ...
    'Fg','RPM','Fpaa','Fs','Fa','Fb','Fc','Foil','Fh','Fw', ...
    'pressure','Fremoved'};
all_fields = [state_fields, input_fields];

t_ref = Xref.(state_fields{1}).t(:);
N = numel(t_ref);
data = zeros(N, 1 + numel(all_fields));
data(:,1) = t_ref;
headers = cell(1, 1 + numel(all_fields)); headers{1} = 'time_h';
units   = cell(1, 1 + numel(all_fields)); units{1} = 'h';

for i = 1:numel(all_fields)
    f = all_fields{i};
    if ~isfield(Xref, f)
        warning('Field %s missing; filling NaN.', f);
        col = nan(N,1); u = '';
    else
        y_vec = Xref.(f).y(:);
        if numel(y_vec) ~= N
            tmp = nan(N,1);
            tmp(1:min(N,numel(y_vec))) = y_vec(1:min(N,numel(y_vec)));
            y_vec = tmp;
        end
        col = y_vec;
        if isfield(Xref.(f), 'yUnit'); u = Xref.(f).yUnit; else; u = ''; end
    end
    data(:, i+1) = col;
    headers{i+1} = f;
    units{i+1}   = u;
end

states_csv = fullfile(out_dir, sprintf('batch_seed%d_b01_states.csv', seed_label));
fid = fopen(states_csv, 'w');
fprintf(fid, '%s\n', strjoin(headers, ','));
fprintf(fid, '%s\n', strjoin(units,   ','));
fclose(fid);
dlmwrite(states_csv, data, '-append', 'precision', '%.10g');
fprintf('  wrote %s  (%d rows x %d cols)\n', states_csv, size(data,1), size(data,2));

% ---- Metadata -----------------------------------------------------------
meta_json = fullfile(out_dir, sprintf('batch_seed%d_b01_meta.json', seed_label));
fid = fopen(meta_json, 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "seed": %d,\n', seed_label);
fprintf(fid, '  "batch_index": 1,\n');
fprintf(fid, '  "faults": %d,\n', faults_override);
fprintf(fid, '  "raman_spec": %d,\n', raman_override);
fprintf(fid, '  "fixed_length": %d,\n', fixed_length_override);
fprintf(fid, '  "n_samples": %d,\n', N);
fprintf(fid, '  "sample_period_h": %.6g,\n', t_ref(2)-t_ref(1));
fprintf(fid, '  "T": %d,\n', T);
fprintf(fid, '  "matlab_version": "%s"\n', strrep(version, '"', ''));
fprintf(fid, '}\n');
fclose(fid);
fprintf('  wrote %s\n\n', meta_json);
