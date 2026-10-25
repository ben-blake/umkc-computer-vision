%hw3_export - Write the Homework 3 numbers to JSON for the report
%   Loads the four results files and writes results/hw3_numbers.json so
%   the figures quoted in the write-up are taken straight from the run
%   instead of being copied by hand.
%
%   See also hw3_q1_features, hw3_q2_gmm, hw3_q3_fisher, hw3_q4_vlad,
%   jsonencode

%   Homework 3 - report export.

clear; clc;
cfg = hw3_config();
q1 = load("hw3_q1_results.mat");
q2 = load("hw3_q2_results.mat");
q3 = load("hw3_q3_results.mat");
q4 = load("hw3_q4_results.mat");

out = struct();
out.types = cfg.types;
out.typeNames = cfg.typeNames;
out.kdList = cfg.kdList;
out.ncList = cfg.ncList;
out.perImage = cfg.perImage;

out.q1 = struct("countsMin", min(q1.counts), "countsMedian", median(q1.counts), ...
    "countsMean", mean(q1.counts), "countsMax", max(q1.counts), ...
    "countsTotal", sum(q1.counts), "seconds", q1.seconds, ...
    "siftByClass", q1.siftByClass);
out.q2 = struct("explained", q2.explained, "gmmStats", q2.gmmStats, ...
    "fitSeconds", q2.fitSeconds, "iterations", q2.iterations, ...
    "converged", q2.converged);

% Accuracy as nested lists, one per feature: rows kd, columns nc.
for t = 1:numel(cfg.types)
    out.q3.(cfg.types(t)) = squeeze(q3.accFV(t, :, :));
    out.q4.(cfg.types(t)) = squeeze(q4.accVLAD(t, :, :));
end
out.q3.encodeSeconds = q3.encodeSeconds;
out.q3.accRaw = q3.accRaw;
out.q3.best = q3.best;
out.q3.mcnemar = q3.mcnemar;
out.q3.pairs = q3.pairs;
out.q3.perClass = q3.perClass;
out.q3.within = q3.within;
out.q3.mcWithin = q3.mcWithin;
out.q3.anyRight = mean(q3.anyRight);
out.q4.accNorm = q4.accNorm;
out.q4.norms = q4.norms;
out.q4.diffFV = q4.diffFV;
out.q4.pFV = q4.pFV;

txt = jsonencode(out, PrettyPrint=true);
fid = fopen(fullfile(cfg.resultsDir, "hw3_numbers.json"), "w");
fwrite(fid, txt); fclose(fid);
fprintf("wrote results/hw3_numbers.json\n");
