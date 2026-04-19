%% matlab_dump_paa_pred.m
% Loads the existing batch_seed42_b01_raman.csv (which holds 1085 captured
% spectra), runs Substrate_prediction's exact PLS + 3-point smoothing logic
% over each sample, and dumps the resulting PAA_pred(1..N) trajectory.
%
% This is the validation reference for indpensim/pat/substrate.py — without
% it, we have no MATLAB ground truth for the PLS prediction chain (batch 1
% uses Raman_spec=1, which never calls Substrate_prediction during the
% normal sim run).
%
% Usage:
%   1. addpath(genpath(pwd))      % so PAA_PLS_model.mat is found
%   2. matlab_dump_paa_pred
%   3. data/matlab_reference/batch_seed42_b01_paa_pred.csv appears.
%
% This script does NOT re-run the full simulation. It just replays the PLS
% step over the captured spectra, which is fast (a few seconds).

clearvars
out_dir = fullfile(pwd, 'data', 'matlab_reference');
in_csv  = fullfile(out_dir, 'batch_seed42_b01_raman.csv');
out_csv = fullfile(out_dir, 'batch_seed42_b01_paa_pred.csv');

if ~exist(in_csv, 'file')
    error('missing %s — run matlab_dump_reference.m first', in_csv);
end

% Read the spectra. Header row is 'wavelength_cm-1', then sample_1..N
% columns. csvread skips the header. (readmatrix handles it automatically.)
M = readmatrix(in_csv);                 % shape (2200, 1+N)
wavelength = M(:, 1);                    % first column = wavenumber axis
Spectra    = M(:, 2:end);                % shape (2200, N), one column per sample
N = size(Spectra, 2);

load PAA_PLS_model                       % brings in `b` (10x212)

PAA_pred = zeros(N, 1);

% Mirror Substrate_prediction.m exactly. Note `j = k - 1` lag in the
% original (predict at k from spectrum at k-1). Because we already have all
% spectra, we just iterate over j directly: for j = 1..N, predict from
% Spectra(:, j) and store at PAA_pred(j).
options1 = [];
No_LV = 4;
for j = 1:N
    Raman_Spec_sg   = sgolayfilt(Spectra(:, j)', 2, 5);
    Raman_Spec_sg   = Raman_Spec_sg';
    Raman_Spec_sg_d = diff(Raman_Spec_sg);
    PAA_peaks_Spec  = Raman_Spec_sg_d([350:500 800:860], :);
    PAA_pred(j)     = PAA_peaks_Spec' * b(No_LV, :)';
    if j > 20
        PAA_pred(j) = (PAA_pred(j-1) + PAA_pred(j-2) + PAA_pred(j)) / 3;
    end
end

% Write CSV with header for easy pandas load on the Python side.
fid = fopen(out_csv, 'w');
fprintf(fid, 'sample,PAA_pred\n');
for j = 1:N
    fprintf(fid, '%d,%.10g\n', j, PAA_pred(j));
end
fclose(fid);

fprintf('Wrote %s (%d samples)\n', out_csv, N);
fprintf('  PAA_pred range: [%.3f, %.3f]\n', min(PAA_pred), max(PAA_pred));
