%% matlab_dump_reference.m
% Dumps a complete IndPenSim batch trajectory to CSV files for use as a
% golden reference during the Python port.
%
% Usage:
%   1. Open MATLAB in the project root.
%   2. Add this folder + the project root to the path:
%        addpath(genpath(pwd))
%   3. Run with reproducible seed:
%        rng(42)
%        Generate_Production_Batch_data_V4
%   4. After it finishes (Raw_Batch_data should be in the workspace), run:
%        matlab_dump_reference
%   5. Three files appear under data/matlab_reference/ :
%        batch_<seed>_<idx>_states.csv     -- 33 ODE states + 12 control inputs, 1 row per sample
%        batch_<seed>_<idx>_raman.csv      -- 2200 Raman wavenumbers x N samples (only if Raman recorded)
%        batch_<seed>_<idx>_meta.json      -- seed, batch flags, sample count, units
%
% Configure below: which batch to dump, and the seed label used in filenames.

%% --- CONFIG -------------------------------------------------------------
seed_label   = 42;          % must match rng(seed) used before generating
batch_index  = 1;           % which batch from Raw_Batch_data to dump
out_dir      = fullfile(pwd, 'data', 'matlab_reference');
%% -----------------------------------------------------------------------

if ~exist(out_dir, 'dir'); mkdir(out_dir); end
if ~exist('Raw_Batch_data', 'var')
    error(['Raw_Batch_data not in workspace. Run ' ...
           'Generate_Production_Batch_data_V4 first.']);
end

batch_ref = sprintf('Batch_%02d', batch_index);
if ~isfield(Raw_Batch_data, batch_ref)
    error('Batch %s not present in Raw_Batch_data.', batch_ref);
end
X = Raw_Batch_data.(batch_ref);

% State + input fields (ODE outputs y(1)..y(33) + the 12 manipulated inputs).
% Field order here matches the y-vector ordering documented in
% docs/state_vector.md, then appends the inputs.
state_fields = { ...
    'S','DO2','O2','P','V','Wt','pH','T','Q','Viscosity', ...
    'Culture_age','a0','a1','a3','a4', ...
    'n0','n1','n2','n3','n4','n5','n6','n7','n8','n9', ...
    'nm','phi0','CO2outgas','CO2_d','PAA','NH3', ...
    'mu_P_calc','mu_X_calc'};   % 33 entries

input_fields = { ...
    'Fg','RPM','Fpaa','Fs','Fa','Fb','Fc','Foil','Fh','Fw', ...
    'pressure','Fremoved'};      % 12 entries

all_fields = [state_fields, input_fields];

% Build the time vector from the first available trace, then per-column
% values. Every X.<field> is an indpensim "channel" with .t, .y, .name, .yUnit.
t_ref = X.(state_fields{1}).t(:);     % column
N = numel(t_ref);

data = zeros(N, 1 + numel(all_fields));
data(:, 1) = t_ref;
headers = cell(1, 1 + numel(all_fields));
headers{1} = 'time_h';
units   = cell(1, 1 + numel(all_fields));
units{1} = 'h';

for i = 1:numel(all_fields)
    f = all_fields{i};
    if ~isfield(X, f)
        warning('Field %s missing from batch %s; filling NaN.', f, batch_ref);
        col = nan(N, 1);
        u = '';
    else
        y_vec = X.(f).y(:);
        if numel(y_vec) ~= N
            warning('Field %s has %d samples vs reference %d; padding/truncating.', ...
                    f, numel(y_vec), N);
            tmp = nan(N, 1);
            tmp(1:min(N, numel(y_vec))) = y_vec(1:min(N, numel(y_vec)));
            y_vec = tmp;
        end
        col = y_vec;
        if isfield(X.(f), 'yUnit'); u = X.(f).yUnit; else; u = ''; end
    end
    data(:, i+1) = col;
    headers{i+1} = f;
    units{i+1}   = u;
end

states_csv = fullfile(out_dir, sprintf('batch_seed%d_b%02d_states.csv', ...
                                        seed_label, batch_index));
fid = fopen(states_csv, 'w');
fprintf(fid, '%s\n', strjoin(headers, ','));   % header row 1: field names
fprintf(fid, '%s\n', strjoin(units,   ','));   % header row 2: units
fclose(fid);
dlmwrite(states_csv, data, '-append', 'precision', '%.10g');
fprintf('Wrote %s  (%d rows x %d cols)\n', states_csv, size(data,1), size(data,2));

% --- Raman spectra (optional) -------------------------------------------
if isfield(X, 'Raman_Spec') && isfield(X.Raman_Spec, 'Intensity') ...
        && ~isempty(X.Raman_Spec.Intensity)
    raman_csv = fullfile(out_dir, sprintf('batch_seed%d_b%02d_raman.csv', ...
                                           seed_label, batch_index));
    wavelengths = X.Raman_Spec.Wavelength(:);   % (W,1)
    intensities = X.Raman_Spec.Intensity;       % (W, N_samples)
    raman_mat = [wavelengths, intensities];
    raman_headers = ['wavelength_cm-1', cellstr(num2str((1:size(intensities,2))', ...
                                                         'sample_%d'))'];
    fid = fopen(raman_csv, 'w');
    fprintf(fid, '%s\n', strjoin(raman_headers, ','));
    fclose(fid);
    dlmwrite(raman_csv, raman_mat, '-append', 'precision', '%.10g');
    fprintf('Wrote %s  (%d wavenumbers x %d samples)\n', ...
            raman_csv, size(intensities,1), size(intensities,2));
end

% --- Metadata -----------------------------------------------------------
meta_json = fullfile(out_dir, sprintf('batch_seed%d_b%02d_meta.json', ...
                                       seed_label, batch_index));
meta = struct( ...
    'seed', seed_label, ...
    'batch_index', batch_index, ...
    'batch_ref', batch_ref, ...
    'n_samples', N, ...
    'sample_period_h', t_ref(2)-t_ref(1), ...
    'state_fields', {state_fields}, ...
    'input_fields', {input_fields}, ...
    'has_raman', isfield(X, 'Raman_Spec'), ...
    'matlab_version', version);
% Crude JSON write (avoid jsonencode dependency on older MATLAB)
fid = fopen(meta_json, 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "seed": %d,\n', meta.seed);
fprintf(fid, '  "batch_index": %d,\n', meta.batch_index);
fprintf(fid, '  "batch_ref": "%s",\n', meta.batch_ref);
fprintf(fid, '  "n_samples": %d,\n', meta.n_samples);
fprintf(fid, '  "sample_period_h": %.6g,\n', meta.sample_period_h);
fprintf(fid, '  "has_raman": %d,\n', meta.has_raman);
fprintf(fid, '  "matlab_version": "%s"\n', strrep(meta.matlab_version, '"', ''));
fprintf(fid, '}\n');
fclose(fid);
fprintf('Wrote %s\n', meta_json);

fprintf('\nDone. Files under %s\n', out_dir);
