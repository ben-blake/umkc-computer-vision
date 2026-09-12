%hw1_part2_export - Write the Part 2 numbers to JSON for the report
%   Loads hw1_part2_results.mat and writes results/part2_numbers.json so
%   the figures quoted in the write-up are taken straight from the run
%   instead of being copied by hand.
%
%   See also hw1_part2_homography, jsonencode

%   Homework 1, Part 2 - report export.

clear; clc;
load hw1_part2_results.mat res M tau sweep pairs

out = struct();
out.M = M; out.tau = tau; out.sweep = sweep;
for i = 1:numel(res)
    r = res(i);
    sz = size(imread(pairs(r.pair, 1)));
    e = struct();
    e.pair = r.pair;
    e.which = char(r.which);
    e.imgA = char(pairs(r.pair,1));
    e.imgB = char(pairs(r.pair,2));
    e.nKey = r.nKey;
    e.nMatch = r.nMatch;
    e.nConsistent = r.nConsistent;
    e.sv = r.sv(:)';
    e.s9 = r.sv(9);
    e.ratio98 = r.sv(9)/r.sv(8);
    e.ratio91 = r.sv(9)/r.sv(1);
    e.rankA = r.rankA;
    e.condA = r.condA;
    e.h = r.h(:)';
    e.H = r.H;
    e.residual = norm(r.A * r.h);
    e.maxErr8 = r.maxErr8;
    e.agree = r.agree;
    e.agreePct = 100*r.agree/r.nMatch;
    e.medianErr = median(r.err);
    e.sweepCounts = r.sweep(:)';
    e.nDistinct = size(unique(r.b1, "rows"), 1);
    e.spreadX = 100*range(r.b1(:,1))/sz(2);
    e.spreadY = 100*range(r.b1(:,2))/sz(1);
    out.entries(i) = e;
end
txt = jsonencode(out, PrettyPrint=true);
fid = fopen("results/part2_numbers.json", "w");
fwrite(fid, txt); fclose(fid);
fprintf("wrote results/part2_numbers.json (%d entries)\n", numel(res));
