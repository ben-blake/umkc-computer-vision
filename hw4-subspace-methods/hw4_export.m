%hw4_export - Write the Homework 4 numbers to JSON for the report
%   Loads the results files and writes results/hw4_numbers.json so the
%   figures quoted in the write-up are taken straight from the run
%   instead of being copied by hand.
%
%   See also hw4_q1_models, hw4_q2_recognition, hw4_q3_retrieval,
%   hw4_q4_local, jsonencode

%   Homework 4 - report export.

clear; clc;
cfg = hw4_config();
F = load(cfg.facesFile, "ids", "selfNames", "celebNames", "faces", "selfFaces", "celebFaces");
M = load("hw4_models.mat", "eig0", "eig1", "eig2", "trainIdx", "h", "W");
q2 = load("hw4_q2_results.mat");
q3 = load("hw4_q3_results.mat");
q4 = load("hw4_q4_results.mat");

out = struct();
out.config = rmfield(cfg, ["dataFile" "selfDir" "celebDir" "facesFile" "resultsDir"]);
out.data = struct("nFaces", size(F.faces, 1), "nSubjects", numel(unique(F.ids)), ...
    "selfNames", F.selfNames, "celebNames", F.celebNames, ...
    "meanBrightness", [mean(F.faces(1:6680, :), "all"), mean(F.celebFaces, "all"), ...
    mean(F.selfFaces, "all")]);
explained = cumsum(M.eig0) / sum(M.eig0);
out.q1 = struct("nTrain", numel(M.trainIdx), "nTrainSubjects", numel(unique(F.ids(M.trainIdx))), ...
    "explained", explained(cfg.d0Sweep), "lda", M.eig1, "lpp", M.eig2, "h", M.h, ...
    "nnzW", nnz(M.W), "sizeW", size(M.W, 1));
out.q2 = struct("acc", q2.acc, "celebAcc", q2.celebAcc, "mcnemar", q2.mcnemar, ...
    "pairs", q2.pairs, "accInSet", q2.accInSet, "d0Sweep", cfg.d0Sweep, ...
    "accSweep", q2.accSweep, "raw", q2.raw, "dims", q2.dims, "accDims", q2.accDims, ...
    "accCosine", q2.accCosine, "nTest", numel(q2.testIdx), ...
    "nTestSubjects", numel(unique(F.ids(q2.testIdx))));
out.q3 = struct("retrieved", q3.retrieved, "apK", q3.apK, "mapK", q3.mapK, ...
    "apKSlide", q3.apKSlide, "mapKSlide", q3.mapKSlide, "mapStdSlide", q3.mapStdSlide, ...
    "recallK", q3.recallK, "mapFull", q3.mapFull, "queries", q3.qNames, ...
    "meanRank", q3.meanRank, "spread", q3.spread, "mapStd", q3.mapStd, ...
    "meanRankStd", q3.meanRankStd);
out.q4 = struct("queries", q4.q_offs, "querySubjects", F.ids(q4.q_offs)', ...
    "accE", q4.accE, "accF", q4.accF, "accG", q4.accG, ...
    "accManyE", q4.accManyE, "accManyF", q4.accManyF, "accManyG", q4.accManyG, ...
    "mcManyEF", q4.mcManyEF, "reachable", q4.reachable, "leads", q4.leads, ...
    "dEG", squeeze(mean(q4.dEG, 1)), "dFG", squeeze(mean(q4.dFG, 1)), ...
    "dEF", squeeze(mean(q4.dEF, 1)), "thetaEG", squeeze(mean(q4.thetaEG, 1)));
% Mean distance between queries, same subject against different, per K
% and per number of leading directions.
same = q4.sameSubj;
other = ~same & ~eye(size(same));
for name = ["dEE" "dFF"]
    D = q4.(name);
    s = zeros(size(D, 3), size(D, 4));
    o = s;
    for k = 1:size(D, 3)
        for l = 1:size(D, 4)
            A = D(:, :, k, l);
            s(k, l) = mean(A(same));
            o(k, l) = mean(A(other));
        end
    end
    out.q4.(name + "Same") = s;
    out.q4.(name + "Other") = o;
end

txt = jsonencode(out, PrettyPrint=true);
fid = fopen(fullfile(cfg.resultsDir, "hw4_numbers.json"), "w");
fwrite(fid, txt); fclose(fid);
fprintf("wrote results/hw4_numbers.json\n");
