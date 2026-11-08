%hw4_q1_models - Train the Eigenface, Fisherface and Laplacianface models
%   Trains A0 = getEigenface, A1 = getFisherface and A2 = getLaplacianface
%   at d0 = cfg.d0 on every subject that is not in the first cfg.nRecog
%   faces, which are the test faces of Q2 and Q3, and not the celebrity.
%   Draws the first six basis images of each model, the PCA spectrum, and
%   the affinity W of the Laplacianface model. Saves the models to
%   hw4_models.mat.
%
%   See also getEigenface, getFisherface, getLaplacianface, hw4_q2_recognition

%   Homework 4, Q1.

clear; clc;
cfg = hw4_config();
load(cfg.facesFile, "faces", "ids");

% Training subjects are disjoint from the test subjects, so the models
% are judged on people they have never seen. The first 800 faces end on
% the last face of subject 110, so no subject is split.
testIds = [unique(ids(1:cfg.nRecog)); cfg.celebId];
trainIdx = find(~ismember(ids, testIds));
train = struct("faces", faces(trainIdx, :), "ids", ids(trainIdx));
fprintf("training: %d faces of %d subjects\n", numel(trainIdx), numel(unique(train.ids)));

tStart = tic; [A0, eig0] = getEigenface(train, cfg.d0); t0 = toc(tStart);
tStart = tic; [A1, eig1] = getFisherface(train, A0); t1 = toc(tStart);
tStart = tic; [A2, W, eig2, h] = getLaplacianface(train, A0); t2 = toc(tStart);
fprintf("A0 %d x %d (%.2f s), A1 %d x %d (%.2f s), A2 %d x %d (%.2f s)\n", ...
    size(A0), t0, size(A1), t1, size(A2), t2);
fprintf("W: %d x %d, %d nonzero (%.2f%%), values %.3f to %.3f\n", size(W), nnz(W), ...
    100 * nnz(W) / numel(W), min(nonzeros(W)), max(nonzeros(W)));

explained = cumsum(eig0) / sum(eig0);
fprintf("PCA variance kept at d0 = 8, 16, 32, 64, 128: %s\n", ...
    mat2str(round(100 * explained(cfg.d0Sweep)', 1)));
fprintf("LDA eigenvalues, first 6: %s, last: %.3g\n", mat2str(eig1(1:6)', 3), eig1(end));
fprintf("LPP eigenvalues, first 6: %s, last: %.3g\n", mat2str(eig2(1:6)', 3), eig2(end));

% The basis images in pixel space: the columns of A0, A0*A1 and A0*A2.
bases = {A0, A0 * A1, A0 * A2};
names = ["eigf", "fisherf", "lapf"];
fig = figure(Name="Face model bases", Position=[100 100 1100 600]);
tl = tiledlayout(fig, 3, 6, TileSpacing="compact", Padding="compact");
for m = 1:3
    for k = 1:6
        ax = nexttile(tl);
        imagesc(ax, reshape(bases{m}(:, k), 20, 20));
        colormap(ax, "gray"); axis(ax, "image", "off");
        title(ax, sprintf("%s_%d", names(m), k));
    end
end
title(tl, sprintf("Q1: first six basis images, d0 = %d, trained on %d faces of %d subjects", ...
    cfg.d0, numel(trainIdx), numel(unique(train.ids))));
hw4_save_figure(fig, "q1_basis.png");

fig = figure(Name="Spectra", Position=[100 100 1300 380]);
tl = tiledlayout(fig, 1, 3, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
semilogy(ax, eig0 / sum(eig0), ".-"); grid(ax, "on"); hold(ax, "on");
xline(ax, cfg.d0, ":", "d0 = " + cfg.d0);
xlabel(ax, "principal component"); ylabel(ax, "share of variance");
title(ax, sprintf("PCA: %.1f%% of the variance in %d components", ...
    100 * explained(cfg.d0), cfg.d0));
ax = nexttile(tl);
stem(ax, eig1, "."); grid(ax, "on");
xlabel(ax, "Fisherface direction"); ylabel(ax, "S_b / S_w along it");
title(ax, "LDA eigenvalues, largest first");
ax = nexttile(tl);
stem(ax, eig2, "."); grid(ax, "on");
xlabel(ax, "Laplacianface direction"); ylabel(ax, "\lambda of X'LX a = \lambda X'DX a");
title(ax, "LPP eigenvalues, smallest first");
title(tl, "Q1: the eigenvalues of the three models");
hw4_save_figure(fig, "q1_spectra.png");

% The affinity, as the course code draws it, for the first 300 training
% faces: distance, heat kernel, same subject, and the supervised W.
nShow = 300;
x = train.faces(1:nShow, :) * A0;
dist = pdist2(x, x);
same = train.ids(1:nShow) == train.ids(1:nShow)';
panels = {dist, exp(-h * dist), same, full(W(1:nShow, 1:nShow))};
ptitles = ["distance ||x_i - x_j||", "heat kernel exp(-h ||x_i - x_j||)", ...
    "same subject", "W: heat kernel within a subject"];
fig = figure(Name="Affinity", Position=[100 100 1100 300]);
tl = tiledlayout(fig, 1, 4, TileSpacing="compact", Padding="compact");
for k = 1:4
    ax = nexttile(tl);
    imagesc(ax, panels{k}); colormap(ax, "gray"); axis(ax, "image");
    title(ax, ptitles(k));
end
title(tl, sprintf("Q1: Laplacianface affinity, first %d training faces", nShow));
hw4_save_figure(fig, "q1_affinity.png");

save hw4_models.mat A0 A1 A2 W h eig0 eig1 eig2 trainIdx testIds
