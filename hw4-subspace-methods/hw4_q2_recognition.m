%hw4_q2_recognition - Top-1 recognition on the first 800 faces
%   Recognizes each of the first cfg.nRecog faces, plus the 10 celebrity
%   faces, by its nearest other face in the Eigenface, Fisherface and
%   Laplacianface spaces of Q1, and draws the three confusion maps. Then
%   repeats it with the models trained on the test faces themselves, which
%   is what the example in the assignment appears to do, sweeps d0, and
%   tests which differences between the models are real.
%
%   See also hw4_q1_models, hw4_loo, getEigenface, getFisherface,
%   getLaplacianface

%   Homework 4, Q2.

clear; clc;
cfg = hw4_config();
load(cfg.facesFile, "faces", "ids");
M = load("hw4_models.mat");

testIdx = [(1:cfg.nRecog)'; find(ids == cfg.celebId)];
X = faces(testIdx, :);
y = ids(testIdx);
nSubj = numel(unique(y));
fprintf("test: %d faces of %d subjects, %d of them the celebrity\n", ...
    numel(y), nSubj, sum(y == cfg.celebId));

names = ["eigenface", "fisherface", "laplacianface"];
proj = {M.A0, M.A0 * M.A1, M.A0 * M.A2};
acc = zeros(1, 3);
cm = cell(1, 3);
pred = cell(1, 3);
for m = 1:3
    [acc(m), cm{m}, pred{m}, labels] = hw4_loo(X * proj{m}, y);
    fprintf("%-14s top-1 accuracy %.3f\n", names(m), acc(m));
end
fprintf("chance %.3f; the most common subject is %.3f of the faces\n", ...
    1 / nSubj, max(accumarray(findgroups(y), 1)) / numel(y));

% The celebrity's 10 faces.
isCeleb = y == cfg.celebId;
celebAcc = cellfun(@(p) mean(p(isCeleb) == cfg.celebId), pred);
fprintf("celebrity recognized: %s of 10\n", mat2str(10 * celebAcc));

fig = figure(Name="Confusion maps", Position=[60 60 1500 520]);
tl = tiledlayout(fig, 1, 3, TileSpacing="compact", Padding="compact");
for m = 1:3
    ax = nexttile(tl);
    imagesc(ax, cm{m} ./ sum(cm{m}, 2), [0 1]);
    colormap(ax, "gray"); axis(ax, "image");
    xlabel(ax, "predicted subject"); ylabel(ax, "true subject");
    title(ax, sprintf("(%d-%d) %s accuracy: %.3f", numel(y), nSubj, names(m), acc(m)));
end
cb = colorbar(ax);
cb.Layout.Tile = "east";
cb.Label.String = "share of the true subject's faces";
title(tl, sprintf("Q2: top-1 confusion maps, d0 = %d, models trained on %d other subjects", ...
    cfg.d0, numel(unique(ids(M.trainIdx)))));
hw4_save_figure(fig, "q2_confusion.png");

% McNemar's exact test between each pair of models on the same faces.
pairs = nchoosek(1:3, 2);
mcnemar = zeros(size(pairs, 1), 3);
fprintf("\nPaired McNemar tests, n = %d\n", numel(y));
for j = 1:size(pairs, 1)
    a = pred{pairs(j, 1)} == y;
    b = pred{pairs(j, 2)} == y;
    onlyA = sum(a & ~b);
    onlyB = sum(~a & b);
    p = min(1, 2 * binocdf(min(onlyA, onlyB), onlyA + onlyB, 0.5));
    mcnemar(j, :) = [onlyA, onlyB, p];
    fprintf("  %-14s vs %-14s %3d vs %3d, p = %.2g\n", names(pairs(j, 1)), ...
        names(pairs(j, 2)), onlyA, onlyB, p);
end

% The same faces with the models trained on the test faces themselves.
% Each face is then part of the training set its own recognition uses,
% and Fisherface and Laplacianface have seen its label.
inSet = struct("faces", X, "ids", y);
B0 = getEigenface(inSet, cfg.d0);
B1 = getFisherface(inSet, B0);
B2 = getLaplacianface(inSet, B0);
accInSet = [hw4_loo(X * B0, y), hw4_loo(X * B0 * B1, y), hw4_loo(X * B0 * B2, y)];
fprintf("\nmodels trained on the test faces: %s\n", mat2str(accInSet, 3));

% Accuracy against d0, models trained on the other subjects.
train = struct("faces", faces(M.trainIdx, :), "ids", ids(M.trainIdx));
accSweep = zeros(numel(cfg.d0Sweep), 3);
for k = 1:numel(cfg.d0Sweep)
    A0 = getEigenface(train, cfg.d0Sweep(k));
    A1 = getFisherface(train, A0);
    A2 = getLaplacianface(train, A0);
    accSweep(k, :) = [hw4_loo(X * A0, y), hw4_loo(X * A0 * A1, y), hw4_loo(X * A0 * A2, y)];
end
fprintf("\naccuracy against d0\n");
disp(array2table(accSweep, RowNames="d0 = " + string(cfg.d0Sweep), VariableNames=names));
raw = hw4_loo(X, y);
fprintf("raw 400 pixels, no model: %.3f\n", raw);

% At d0 = 32, keep only the first d directions of each model. The
% Fisherface directions are ordered by Sb/Sw and the Laplacianface ones
% by how well they keep a subject together, so the first few should
% carry the most.
dims = [4 8 12 16 24 32];
accDims = zeros(numel(dims), 3);
for k = 1:numel(dims)
    d = dims(k);
    accDims(k, :) = [hw4_loo(X * M.A0(:, 1:d), y), ...
        hw4_loo(X * M.A0 * M.A1(:, 1:d), y), hw4_loo(X * M.A0 * M.A2(:, 1:d), y)];
end
fprintf("\naccuracy against directions kept, d0 = %d\n", cfg.d0);
disp(array2table(accDims, RowNames="d = " + string(dims), VariableNames=names));

% The same spaces compared by cosine distance, which ignores the length
% of a face vector, that is its overall brightness and contrast.
accCosine = zeros(1, 3);
for m = 1:3
    Z = X * proj{m};
    accCosine(m) = hw4_loo(Z ./ vecnorm(Z, 2, 2), y);
end
fprintf("cosine distance instead of Euclidean, d0 = %d: %s\n", cfg.d0, mat2str(accCosine, 3));

fig = figure(Name="Accuracy against dimension", Position=[100 100 1100 420]);
tl = tiledlayout(fig, 1, 2, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
semilogx(ax, cfg.d0Sweep, accSweep, "-o", LineWidth=1.4); hold(ax, "on");
yline(ax, raw, ":", "raw pixels", LabelHorizontalAlignment="left");
grid(ax, "on"); xticks(ax, cfg.d0Sweep);
xlabel(ax, "d0, Eigenface dimension, all directions kept"); ylabel(ax, "top-1 accuracy");
legend(ax, names, Location="southeast");
title(ax, "against d0");
ax = nexttile(tl);
plot(ax, dims, accDims, "-o", LineWidth=1.4); hold(ax, "on");
yline(ax, raw, ":", "raw pixels", LabelHorizontalAlignment="left");
grid(ax, "on"); xticks(ax, dims);
xlabel(ax, sprintf("directions kept, d0 = %d", cfg.d0)); ylabel(ax, "top-1 accuracy");
legend(ax, names, Location="southeast");
title(ax, "against the number of leading directions");
title(tl, sprintf("Q2: accuracy on the %d test faces against dimension", numel(y)));
hw4_save_figure(fig, "q2_dimension.png");

save hw4_q2_results.mat acc cm pred labels testIdx celebAcc mcnemar pairs ...
    accInSet accSweep raw dims accDims accCosine
