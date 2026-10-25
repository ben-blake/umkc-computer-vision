%hw3_q4_vlad - VLAD recognition for every feature and GMM size
%   Encodes the 600 images with getVLAD, using the means of the same 18
%   GMMs as k-means centroids, classifies them by leave-one-out 1-NN with
%   knnclassify, and fills the extra credit tables. Draws the six
%   confusion maps of each feature, tries intra-normalization, and
%   compares VLAD with the Fisher vector on the same models.
%
%   See also getVLAD, hw3_loo, hw3_confusion_figure, hw3_q3_fisher

%   Homework 3, Q4 - extra credit.

clear; clc;
cfg = hw3_config();
load gradient_features.mat f_sift f_dsift f_hog
load gmm_models.mat models A
q3 = load("hw3_q3_results.mat", "accFV", "predFV");
allFeats = {f_sift, f_dsift, f_hog};
N = numel(cfg.files);
nType = numel(cfg.types);
nKd = numel(cfg.kdList);
nNc = numel(cfg.ncList);
y = cfg.labels;

accVLAD = zeros(nType, nKd, nNc);
cmVLAD = cell(nType, nKd, nNc);
predVLAD = cell(nType, nKd, nNc);
for t = 1:nType
    for a = 1:nKd
        kd = cfg.kdList(a);
        % getVLAD takes features that are already reduced.
        reduced = cellfun(@(f) double(f) * A{t}(:, 1:kd), allFeats{t}, ...
            UniformOutput=false);
        for b = 1:nNc
            nc = cfg.ncList(b);
            X = zeros(N, kd*nc);
            for i = 1:N
                X(i, :) = getVLAD(reduced{i}, models{t, a, b}, kd, nc);
            end
            [accVLAD(t, a, b), cmVLAD{t, a, b}, predVLAD{t, a, b}] = hw3_loo(X, y);
            fprintf("%-11s kd = %2d, nc = %3d, D = %5d: %.3f\n", ...
                cfg.typeNames(t), kd, nc, kd*nc, accVLAD(t, a, b));
        end
    end
end

for t = 1:nType
    fprintf("\n%s VLAD aggregation recognition accuracy\n", cfg.typeNames(t));
    disp(array2table(squeeze(accVLAD(t, :, :)), ...
        RowNames="kd = " + string(cfg.kdList), ...
        VariableNames="nc = " + string(cfg.ncList)));
end

for t = 1:nType
    fig = hw3_confusion_figure(cmVLAD, accVLAD, cfg, t, "vlad");
    drawnow;
    exportgraphics(fig, fullfile(cfg.resultsDir, ...
        sprintf("q4_confusion_%s.png", cfg.types(t))), Resolution=150);
end

% Normalization of VLAD at kd = 16 and nc = 64.
a = 1; b = 2;
norms = ["none" "intra" "improved"];
accNorm = zeros(nType, numel(norms));
for t = 1:nType
    kd = cfg.kdList(a); nc = cfg.ncList(b);
    for m = 1:numel(norms)
        X = zeros(N, kd*nc);
        for i = 1:N
            X(i, :) = getVLAD(double(allFeats{t}{i}) * A{t}(:, 1:kd), ...
                models{t, a, b}, kd, nc, Normalization=norms(m));
        end
        accNorm(t, m) = hw3_loo(X, y);
    end
end
fprintf("\nVLAD normalization at kd = %d, nc = %d\n", cfg.kdList(a), cfg.ncList(b));
disp(array2table(accNorm, RowNames=cfg.typeNames, VariableNames=norms));

% VLAD against the Fisher vector on the same 18 models, paired over the
% same 600 images.
diffFV = q3.accFV - accVLAD;
fprintf("\nFisher vector minus VLAD, all 18 models: mean %+.3f, range %+.3f to %+.3f\n", ...
    mean(diffFV(:)), min(diffFV(:)), max(diffFV(:)));
% McNemar test model by model. The 18 tests share images, so they are
% not independent, and each one is reported on its own.
pFV = zeros(size(predVLAD));
for j = 1:numel(predVLAD)
    onlyFV = sum(q3.predFV{j} == y & predVLAD{j} ~= y);
    onlyVLAD = sum(q3.predFV{j} ~= y & predVLAD{j} == y);
    pFV(j) = min(1, 2 * binocdf(min(onlyFV, onlyVLAD), onlyFV + onlyVLAD, 0.5));
end
fprintf("FV better than VLAD at p < 0.05 in %d of %d models, p from %.2g to %.2g\n", ...
    sum(pFV(:) < 0.05 & diffFV(:) > 0), numel(pFV), min(pFV(:)), max(pFV(:)));

save hw3_q4_results.mat accVLAD cmVLAD predVLAD accNorm norms diffFV pFV
