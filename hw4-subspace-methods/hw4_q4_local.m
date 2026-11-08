%hw4_q4_local - KNN-driven local Eigenface and Fisherface models
%   Represents every face by a kd = cfg.kdLocal PCA of the data set, draws
%   cfg.nQuery queries at random from the first cfg.nLocal faces with the
%   code in the assignment, and for each query and each K in cfg.knnList
%   finds its K nearest faces with a kd-tree. A local Eigenface model and
%   a local Fisherface model are trained on those K faces only, and the
%   query is recognized as its nearest neighbor among them in each local
%   space. Reports top-1 accuracy and confusion maps, repeats the
%   experiment with more queries for a reliable number, and computes the
%   Cauchy-Binet Grassmann distance between the local models.
%
%   See also getEigenface, getFisherface, getGrassmannDistance, knnsearch

%   Homework 4, Q4 - extra credit.

clear; clc;
cfg = hw4_config();
load(cfg.facesFile, "faces", "ids");
nK = numel(cfg.knnList);

% The kd = 16 representation, from a PCA of the whole data set.
Ag = getEigenface(faces, cfg.kdLocal);
z = faces * Ag;
searcher = KDTreeSearcher(z(1:cfg.nLocal, :));

% The query draw from the assignment, seeded so it can be repeated.
rng(cfg.seed);
q_offs = randperm(cfg.nLocal);
q_offs = q_offs(1:cfg.nQuery);
fprintf("queries: %s\n", mat2str(q_offs));
fprintf("their subjects: %s\n", mat2str(ids(q_offs)'));

[predE, predF, predG, nOwn, bases] = iRun(q_offs, faces, ids, z, searcher, cfg);
accE = mean(predE == ids(q_offs), 1);
accF = mean(predF == ids(q_offs), 1);
accG = mean(predG == ids(q_offs));
fprintf("\n%-26s %s\n", "top-1 accuracy, 20 queries", strjoin("K = " + string(cfg.knnList), "   "));
fprintf("%-26s %s\n", "local Eigenface", sprintf("%7.2f  ", accE));
fprintf("%-26s %s\n", "local Fisherface", sprintf("%7.2f  ", accF));
fprintf("%-26s %7.2f\n", "global kd = 16, 1-NN", accG);
fprintf("faces of the query's own subject among the K: median %s\n", ...
    mat2str(median(nOwn, 1)));

% 20 queries give accuracy in steps of 5 points, so the same experiment
% with 500 queries, the first 20 being the ones above.
rng(cfg.seed);
q_many = randperm(cfg.nLocal, 500);
q_many = [q_offs, setdiff(q_many, q_offs, "stable")];
q_many = q_many(1:500);
[pE, pF, pG, nOwnMany] = iRun(q_many, faces, ids, z, searcher, cfg);
yMany = ids(q_many);
accManyE = mean(pE == yMany, 1);
accManyF = mean(pF == yMany, 1);
accManyG = mean(pG == yMany);
fprintf("\n%-26s %s\n", "top-1 accuracy, 500 queries", strjoin("K = " + string(cfg.knnList), "   "));
fprintf("%-26s %s\n", "local Eigenface", sprintf("%7.3f  ", accManyE));
fprintf("%-26s %s\n", "local Fisherface", sprintf("%7.3f  ", accManyF));
fprintf("%-26s %7.3f\n", "global kd = 16, 1-NN", accManyG);
mcManyEF = zeros(nK, 3);
for k = 1:nK
    a = pE(:, k) == yMany; b = pF(:, k) == yMany;
    onlyF = sum(b & ~a); onlyE = sum(a & ~b);
    mcManyEF(k, :) = [onlyF, onlyE, min(1, 2 * binocdf(min(onlyF, onlyE), onlyF + onlyE, 0.5))];
    fprintf("  K = %d: Fisherface only %d, Eigenface only %d, McNemar p = %.2g\n", ...
        cfg.knnList(k), mcManyEF(k, :));
end
% Could a perfect local model have found the subject at all? Only when a
% face of the query's subject is among its K neighbors.
reachable = mean(nOwnMany > 0, 1);
fprintf("queries with their own subject among the K: %s\n", mat2str(reachable, 3));

% Confusion maps of the 20 queries, over the subjects that appear in them.
fig = figure(Name="Local models", Position=[60 60 1400 820]);
tl = tiledlayout(fig, 2, nK, TileSpacing="compact", Padding="compact");
preds = {predE, predF};
mnames = ["local eigenface", "local fisherface"];
accs = {accE, accF};
y = ids(q_offs)';
for m = 1:2
    for k = 1:nK
        p = preds{m}(:, k)';
        labels = unique([y, p]);
        cmk = confusionmat(y, p, Order=labels);
        ax = nexttile(tl);
        imagesc(ax, cmk, [0 1]); colormap(ax, "gray"); axis(ax, "image");
        xticks(ax, 1:numel(labels)); yticks(ax, 1:numel(labels));
        xticklabels(ax, string(labels)); yticklabels(ax, string(labels));
        xtickangle(ax, 90); ax.FontSize = 7;
        xlabel(ax, "predicted subject"); ylabel(ax, "true subject");
        title(ax, sprintf("%s K = %d accuracy: %.2f", mnames(m), cfg.knnList(k), accs{m}(k)), FontSize=10);
    end
end
title(tl, sprintf("Q4: confusion maps of the %d queries, over the %s", cfg.nQuery, ...
    "subjects that occur as truth or prediction"));
hw4_save_figure(fig, "q4_confusion.png");

% Cauchy-Binet distances. Each query has a local Eigenface subspace and
% a local Fisherface subspace of the 400 pixel space. Between the 20
% queries, and between each local model and the global one.
% Two versions: all 16 directions of each model, and only its leading 4.
% A product of 16 cosines is almost always near 0, so the 16-direction
% distance saturates at 1, and the Fisherface subspace A0*A1 lies inside
% the span of A0, so their 16-direction distance is 0 by construction.
% The leading 4 directions avoid both.
leads = [cfg.kdLocal 4];
nL = numel(leads);
dEE = zeros(cfg.nQuery, cfg.nQuery, nK, nL);
dFF = zeros(cfg.nQuery, cfg.nQuery, nK, nL);
dEG = zeros(cfg.nQuery, nK, nL);
dFG = zeros(cfg.nQuery, nK, nL);
dEF = zeros(cfg.nQuery, nK, nL);
thetaEG = zeros(cfg.nQuery, nK, nL);
for l = 1:nL
    lead = @(B) B(:, 1:min(leads(l), size(B, 2)));
    G = lead(Ag);
    for k = 1:nK
        for i = 1:cfg.nQuery
            Ei = lead(bases.E{i, k});
            Fi = lead(bases.F{i, k});
            [dEG(i, k, l), th] = getGrassmannDistance(Ei, G);
            thetaEG(i, k, l) = rad2deg(mean(th));
            dFG(i, k, l) = getGrassmannDistance(Fi, G);
            dEF(i, k, l) = getGrassmannDistance(Ei, Fi);
            for j = i + 1:cfg.nQuery
                dEE(i, j, k, l) = getGrassmannDistance(Ei, lead(bases.E{j, k}));
                dFF(i, j, k, l) = getGrassmannDistance(Fi, lead(bases.F{j, k}));
            end
        end
        dEE(:, :, k, l) = dEE(:, :, k, l) + dEE(:, :, k, l)';
        dFF(:, :, k, l) = dFF(:, :, k, l) + dFF(:, :, k, l)';
    end
end

% Pairs of queries that are the same subject, against all other pairs.
sameSubj = (ids(q_offs) == ids(q_offs)') & ~eye(cfg.nQuery);
otherSubj = ~sameSubj & ~eye(cfg.nQuery);
pr = @(name, v) fprintf("%-50s %s\n", name, sprintf("%7.3f  ", v));
for l = 1:nL
    fprintf("\nCauchy-Binet distance, leading %d directions, mean over the 20 queries\n", leads(l));
    fprintf("%-50s %s\n", "", strjoin("K = " + string(cfg.knnList), "   "));
    pr("local Eigenface vs global", mean(dEG(:, :, l), 1));
    pr("local Fisherface vs global", mean(dFG(:, :, l), 1));
    pr("local Eigenface vs local Fisherface", mean(dEF(:, :, l), 1));
    for m = 1:2
        D = {dEE, dFF};
        D = D{m}(:, :, :, l);
        pr("between queries, " + mnames(m) + ", same subject", ...
            arrayfun(@(k) iMaskedMean(D(:, :, k), sameSubj), 1:nK));
        pr("between queries, " + mnames(m) + ", different", ...
            arrayfun(@(k) iMaskedMean(D(:, :, k), otherSubj), 1:nK));
    end
    fprintf("mean principal angle, local vs global Eigenface: %s degrees\n", ...
        mat2str(mean(thetaEG(:, :, l), 1), 3));
end

% The figure uses the leading 4 directions, which do not saturate.
l = 2;
order = [find(any(sameSubj, 2)); find(~any(sameSubj, 2))];   % same-subject queries first
fig = figure(Name="Grassmann distances", Position=[60 60 1400 820]);
tl = tiledlayout(fig, 2, nK, TileSpacing="compact", Padding="compact");
dd = {dEE, dFF};
for m = 1:2
    for k = 1:nK
        ax = nexttile(tl);
        imagesc(ax, dd{m}(order, order, k, l), [0 1]); colormap(ax, "parula"); axis(ax, "image");
        xticks(ax, 1:cfg.nQuery); yticks(ax, 1:cfg.nQuery);
        xticklabels(ax, string(ids(q_offs(order)))); yticklabels(ax, string(ids(q_offs(order))));
        xtickangle(ax, 90); ax.FontSize = 7;
        xlabel(ax, "query subject"); ylabel(ax, "query subject");
        title(ax, sprintf("%s models, K = %d", mnames(m), cfg.knnList(k)), FontSize=10);
    end
end
cb = colorbar(ax);
cb.Layout.Tile = "east";
cb.Label.String = "Cauchy-Binet distance, leading 4 directions";
title(tl, "Q4: Grassmann distance between the local models of the 20 queries");
hw4_save_figure(fig, "q4_grassmann.png");

save hw4_q4_results.mat q_offs predE predF predG accE accF accG nOwn ...
    q_many accManyE accManyF accManyG mcManyEF reachable ...
    leads dEE dFF dEG dFG dEF thetaEG sameSubj

function [predE, predF, predG, nOwn, bases] = iRun(queries, faces, ids, z, searcher, cfg)
%iRun - Local models and their predictions for a set of queries
nQ = numel(queries);
nK = numel(cfg.knnList);
predE = zeros(nQ, nK);
predF = zeros(nQ, nK);
predG = zeros(nQ, 1);
nOwn = zeros(nQ, nK);
bases = struct("E", {cell(nQ, nK)}, "F", {cell(nQ, nK)});
for i = 1:nQ
    q = queries(i);
    % K + 1 neighbors, then the query itself is removed.
    nb = knnsearch(searcher, z(q, :), K=max(cfg.knnList) + 1);
    nb = nb(nb ~= q);
    predG(i) = ids(nb(1));
    for k = 1:nK
        idx = nb(1:cfg.knnList(k));
        nOwn(i, k) = sum(ids(idx) == ids(q));
        local = struct("faces", faces(idx, :), "ids", ids(idx));
        A0 = getEigenface(local, cfg.kdLocal);
        A1 = getFisherface(local, A0, Regularization=1e-3);
        predE(i, k) = iNearest(faces(q, :) * A0, local.faces * A0, local.ids);
        predF(i, k) = iNearest(faces(q, :) * A0 * A1, local.faces * A0 * A1, local.ids);
        bases.E{i, k} = A0;
        bases.F{i, k} = A0 * A1;
    end
end
end

function label = iNearest(xq, X, y)
%iNearest - Label of the nearest row of X to xq
[~, j] = min(sum((X - xq).^2, 2));
label = y(j);
end

function m = iMaskedMean(A, mask)
%iMaskedMean - Mean of the entries of A where mask is true
m = mean(A(mask));
end
