%hw3_q3_fisher - Fisher vector recognition for every feature and GMM size
%   Encodes the 600 images as Fisher vectors with getFisherVector for each
%   of the three feature types and the six GMM models of each, classifies
%   them by leave-one-out 1-NN with knnclassify, and fills the accuracy
%   tables of the assignment. Draws the six confusion maps of each
%   feature, checks what the improved normalization contributes, and
%   tests which of the differences between features are real.
%
%   See also getFisherVector, hw3_loo, hw3_confusion_figure, hw3_q4_vlad

%   Homework 3, Q3.

clear; clc;
cfg = hw3_config();
load gradient_features.mat f_sift f_dsift f_hog
load gmm_models.mat models A
allFeats = {f_sift, f_dsift, f_hog};
N = numel(cfg.files);
nType = numel(cfg.types);
nKd = numel(cfg.kdList);
nNc = numel(cfg.ncList);
y = cfg.labels;

accFV = zeros(nType, nKd, nNc);
cmFV = cell(nType, nKd, nNc);
predFV = cell(nType, nKd, nNc);
encodeSeconds = zeros(nType, nKd, nNc);
for t = 1:nType
    for a = 1:nKd
        for b = 1:nNc
            kd = cfg.kdList(a); nc = cfg.ncList(b);
            gmm = models{t, a, b};
            X = zeros(N, 2*kd*nc);
            tStart = tic;
            for i = 1:N
                X(i, :) = getFisherVector(allFeats{t}{i}, A{t}, gmm, kd, nc);
            end
            encodeSeconds(t, a, b) = toc(tStart);
            [accFV(t, a, b), cmFV{t, a, b}, predFV{t, a, b}] = hw3_loo(X, y);
            fprintf("%-11s kd = %2d, nc = %3d, D = %5d: %.3f  (%.1f s)\n", ...
                cfg.typeNames(t), kd, nc, 2*kd*nc, accFV(t, a, b), ...
                encodeSeconds(t, a, b));
        end
    end
end

% The three tables of the assignment. Chance is 1/15 = 0.067.
for t = 1:nType
    fprintf("\n%s FV aggregation recognition accuracy\n", cfg.typeNames(t));
    disp(array2table(squeeze(accFV(t, :, :)), ...
        RowNames="kd = " + string(cfg.kdList), ...
        VariableNames="nc = " + string(cfg.ncList)));
end

for t = 1:nType
    fig = hw3_confusion_figure(cmFV, accFV, cfg, t, "fv");
    drawnow;
    exportgraphics(fig, fullfile(cfg.resultsDir, ...
        sprintf("q3_confusion_%s.png", cfg.types(t))), Resolution=150);
end

% What the improved normalization is worth: the same Fisher vectors with
% no square root and no L2 norm, at kd = 16 and nc = 64.
a = 1; b = 2;
accRaw = zeros(nType, 1);
for t = 1:nType
    X = zeros(N, 2*cfg.kdList(a)*cfg.ncList(b));
    for i = 1:N
        X(i, :) = getFisherVector(allFeats{t}{i}, A{t}, models{t, a, b}, ...
            cfg.kdList(a), cfg.ncList(b), Normalization="none");
    end
    accRaw(t) = hw3_loo(X, y);
end
fprintf("\nNormalization at kd = %d, nc = %d\n", cfg.kdList(a), cfg.ncList(b));
disp(table(cfg.typeNames', accRaw, squeeze(accFV(:, a, b)), ...
    VariableNames=["feature" "none" "improved"]));

% Is the best configuration of one feature really better than the best
% of another? McNemar's exact test on the images one gets right and the
% other gets wrong, all 600 of them, as in Homework 1.
best = zeros(nType, 1);
for t = 1:nType
    [~, best(t)] = max(reshape(accFV(t, :, :), 1, []));
end
pairs = nchoosek(1:nType, 2);
mcnemar = zeros(size(pairs, 1), 3);
fprintf("\nBest configuration of each feature, paired McNemar test, n = %d\n", N);
for j = 1:size(pairs, 1)
    t1 = pairs(j, 1); t2 = pairs(j, 2);
    [a1, b1] = ind2sub([nKd nNc], best(t1));
    [a2, b2] = ind2sub([nKd nNc], best(t2));
    p1 = predFV{t1, a1, b1}; p2 = predFV{t2, a2, b2};
    only1 = sum(p1 == y & p2 ~= y);
    only2 = sum(p1 ~= y & p2 == y);
    p = min(1, 2 * binocdf(min(only1, only2), only1 + only2, 0.5));
    mcnemar(j, :) = [only1, only2, p];
    fprintf("  %-11s (%.3f) vs %-11s (%.3f): %3d vs %3d, p = %.2g\n", ...
        cfg.typeNames(t1), accFV(t1, a1, b1), cfg.typeNames(t2), ...
        accFV(t2, a2, b2), only1, only2, p);
end

% Does kd or nc matter within one feature? Same test, chosen pairs.
within = [2 2 3 2 1 3;      % dense SIFT, kd 32 vs 16, nc = 128
          2 2 1 2 1 1;      % dense SIFT, kd 32 vs 16, nc = 32
          1 1 1 1 1 3;      % SIFT, nc 32 vs 128, kd = 16
          3 1 1 3 2 1];     % HoG, kd 16 vs 32, nc = 32
mcWithin = zeros(size(within, 1), 3);
fprintf("\nWithin one feature, paired McNemar test\n");
for j = 1:size(within, 1)
    w = within(j, :);
    p1 = predFV{w(1), w(2), w(3)}; p2 = predFV{w(4), w(5), w(6)};
    only1 = sum(p1 == y & p2 ~= y);
    only2 = sum(p1 ~= y & p2 == y);
    p = min(1, 2 * binocdf(min(only1, only2), only1 + only2, 0.5));
    mcWithin(j, :) = [only1, only2, p];
    fprintf("  %-11s (%d, %3d) vs (%d, %3d): %3d vs %3d, p = %.2g\n", ...
        cfg.typeNames(w(1)), cfg.kdList(w(2)), cfg.ncList(w(3)), ...
        cfg.kdList(w(5)), cfg.ncList(w(6)), only1, only2, p);
end

% How many images at least one of the three best configurations gets
% right, an upper bound on what combining the features could reach.
anyRight = false(N, 1);
for t = 1:nType
    [a1, b1] = ind2sub([nKd nNc], best(t));
    anyRight = anyRight | predFV{t, a1, b1} == y;
end
fprintf("\nright under at least one feature's best FV: %.3f\n", mean(anyRight));

% Per-class accuracy of the best configuration of each feature.
perClass = zeros(numel(cfg.classes), nType);
for t = 1:nType
    [a1, b1] = ind2sub([nKd nNc], best(t));
    cm = cmFV{t, a1, b1};
    perClass(:, t) = diag(cm) ./ sum(cm, 2);
end
fprintf("\nPer-class accuracy, best configuration of each feature\n");
disp(array2table(100 * perClass, RowNames=cfg.classes, VariableNames=cfg.typeNames));

save hw3_q3_results.mat accFV cmFV predFV encodeSeconds accRaw best mcnemar pairs ...
    perClass within mcWithin anyRight
