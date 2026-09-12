%hw1_part2_figs - Visualize the HSV histogram features
%   Shows one image per class with its K = 64 histogram under both
%   metrics, and the class-mean histograms as a heat map, which is where
%   the separability of the feature becomes visible.
%
%   See also getHSVHist, hw1_part2_hist, hw1_part3_recog

%   Homework 1, Part 2 - figures.

clear; clc;
load hsv_hist64.mat hsv_hist labels classes color_table
hEuc = hsv_hist;
load hsv_hist64_kld.mat hsv_hist
hKld = hsv_hist;
[files, ~, ~] = hw1_config();

% Two classes with very different color statistics.
show = [5 14];                          % beach and forest
figHist = figure(Name="HSV histograms", Position=[100 100 1000 640]);
tl = tiledlayout(figHist, 3, 2, TileSpacing="compact", Padding="compact");
for j = 1:2
    idx = (show(j)-1)*40 + 1;
    ax = nexttile(tl, j);
    imshow(imread(files(idx)), Parent=ax);
    title(ax, classes(show(j)), Interpreter="none");

    ax = nexttile(tl, j+2);
    bar(ax, hEuc(idx, :), 1, FaceColor="flat", CData=hsv2rgb(color_table));
    xlim(ax, [0 65]); ylabel(ax, "frequency"); title(ax, "euc, K = 64");

    ax = nexttile(tl, j+4);
    bar(ax, hKld(idx, :), 1, FaceColor="flat", CData=hsv2rgb(color_table));
    xlim(ax, [0 65]); ylabel(ax, "frequency"); title(ax, "kld, K = 64");
    xlabel(ax, "code word (sorted by hue)");
end
title(tl, "Part 2: HSV histograms of two images");
theme(figHist, "light");
drawnow;
exportgraphics(figHist, "results/part2_histograms.png", Resolution=150);

% Class means. Rows that look alike are the classes k-NN will confuse.
meanEuc = zeros(numel(classes), size(hEuc, 2));
meanKld = zeros(numel(classes), size(hKld, 2));
for k = 1:numel(classes)
    meanEuc(k, :) = mean(hEuc(labels == k, :), 1);
    meanKld(k, :) = mean(hKld(labels == k, :), 1);
end

figMean = figure(Name="Class mean histograms", Position=[100 100 1000 480]);
tl = tiledlayout(figMean, 1, 2, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
imagesc(ax, meanEuc); colorbar(ax);
yticks(ax, 1:numel(classes)); yticklabels(ax, classes);
set(ax, TickLabelInterpreter="none");
xlabel(ax, "code word"); title(ax, "euclidean assignment");
ax = nexttile(tl);
imagesc(ax, meanKld); colorbar(ax);
yticks(ax, 1:numel(classes)); yticklabels(ax, []);
xlabel(ax, "code word"); title(ax, "KL assignment");
title(tl, "Part 2: class-mean HSV histograms, K = 64");
theme(figMean, "light");
drawnow;
exportgraphics(figMean, "results/part2_class_means.png", Resolution=150);

% How concentrated is each histogram, i.e. how many bins are really used.
fprintf("Mean number of non-empty bins: euc %.1f, kld %.1f (of 64)\n", ...
    mean(sum(hEuc > 0, 2)), mean(sum(hKld > 0, 2)));
fprintf("Mean histogram entropy:        euc %.2f, kld %.2f bits\n", ...
    mean(-sum(hEuc .* log2(hEuc + eps), 2)), ...
    mean(-sum(hKld .* log2(hKld + eps), 2)));
