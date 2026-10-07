%hw2_q3_export - Write the Homework 2 numbers to JSON for the report
%   Loads the three results files and writes results/hw2_numbers.json so
%   the figures quoted in the write-up are taken straight from the run
%   instead of being copied by hand.
%
%   See also hw2_q1_pyramid, hw2_q2_extrema, hw2_q3_keypoints, jsonencode

%   Homework 2 - report export.

clear; clc;
q1 = load("hw2_q1_results.mat");
q2 = load("hw2_q2_results.mat");
q3 = load("hw2_q3_results.mat");
cfg = hw2_config();

out = struct();
out.scales = cfg.scales;
out.logPeakThres = cfg.logPeakThres;
out.matchRadius = cfg.matchRadius;
out.edgeRatio = cfg.edgeRatio;
out.thresSweep = cfg.thresSweep;
out.budgets = cfg.budgets;

out.q1 = struct("levelStats", q1.stats, "rawMean", q1.rawMean, ...
    "seconds", q1.elapsed);

examples = rmfield(q2.examples, "name");
for j = 1:numel(examples)
    examples(j).label = char(q2.examples(j).name);
end
out.q2 = struct("examples", examples, "fit", q2.fit);

for n = 1:numel(q3.res)
    r = q3.res(n);
    e = struct();
    e.image = char(r.image);
    e.size = r.size;
    e.nLoG = r.nLoG;
    e.nSift = r.nSift;
    e.nSiftInRange = r.nSiftInRange;
    e.secondsLoG = r.tLoG;
    e.secondsSift = r.tSift;
    e.sweep = r.sweep;
    e.matchAll = r.matchAll;
    e.matchPct = 100 * r.matchAll(1) / r.matchAll(2);
    e.chancePct = 100 * r.matchAll(3) / r.matchAll(2);
    e.matchBack = r.matchBack;
    e.matchBackPct = 100 * r.matchBack(1) / r.matchBack(2);
    e.medianDist = r.medianDist;
    e.matchBudget = r.matchBudget;
    e.scaleRatio = r.scaleRatio;
    e.scaleCorr = r.scaleCorr;
    e.nEdge = r.nEdge;
    e.edgePct = 100 * r.nEdge / r.nLoG;
    e.matchEdge = r.matchEdge;
    e.matchNotEdge = r.matchNotEdge;
    e.logScaleHist = r.logScaleHist;
    e.siftScaleHist = r.siftScaleHist;
    out.q3(n) = e;
end

txt = jsonencode(out, PrettyPrint=true);
fid = fopen(fullfile(cfg.resultsDir, "hw2_numbers.json"), "w");
fwrite(fid, txt); fclose(fid);
fprintf("wrote results/hw2_numbers.json (%d images)\n", numel(q3.res));
