%hw2_q1_pyramid - Build and show the LoG pyramid of the example image
%   Computes the seven-level LoG pyramid of flarsheim-1.jpg with
%   getLoGPyramid, shows the magnitude of every level next to the input,
%   and prints how the response is distributed on each level.
%
%   See also getLoGPyramid, hw2_config, hw2_q2_extrema

%   Homework 2, Q1 - figures.

clear; clc;
cfg = hw2_config();
scales = cfg.scales;
N = numel(scales);

im = imread(cfg.images(1));
tStart = tic;
P = getLoGPyramid(im, scales);
elapsed = toc(tStart);
A = abs(P);

fprintf("image %d by %d, pyramid %d by %d by %d, %.2f s\n", ...
    size(im, 1), size(im, 2), size(P, 1), size(P, 2), size(P, 3), elapsed);
fprintf("scale step between levels: %s\n", ...
    mat2str(scales(2:end) ./ scales(1:end-1), 4));

fprintf("\n%5s %7s %8s %8s %8s %8s\n", ...
    "level", "sigma", "kernel", "mean|L|", "p99|L|", "max|L|");
stats = zeros(N, 3);
for k = 1:N
    a = reshape(A(:, :, k), [], 1);
    stats(k, :) = [mean(a), prctile(a, 99), max(a)];
    fprintf("%5d %7.3f %8d %8.2f %8.2f %8.2f\n", k, scales(k), ...
        2*ceil(4*scales(k)) + 1, stats(k, 1), stats(k, 2), stats(k, 3));
end

% Without the sigma^2 factor the response would shrink with scale.
raw = A ./ reshape(scales.^2, 1, 1, N);
rawMean = squeeze(mean(raw, [1 2]))';
fprintf("\nmean |L| without sigma^2: %s\n", mat2str(rawMean, 3));
fprintf("largest over smallest level: %.1f without, %.2f with\n", ...
    rawMean(1) / rawMean(end), max(stats(:, 1)) / min(stats(:, 1)));

% One color scale for all levels so they can be compared by eye.
cmax = prctile(A(:), 99.5);
fig = figure(Name="LoG pyramid", Position=[100 100 1200 500]);
tl = tiledlayout(fig, 2, 4, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
imshow(im2gray(im), Parent=ax);
title(ax, "input");
for k = 1:N
    ax = nexttile(tl);
    imagesc(ax, A(:, :, k), [0 cmax]);
    axis(ax, "image"); axis(ax, "off"); colormap(ax, "gray");
    title(ax, sprintf("|LoG|, \\sigma = %.3f", scales(k)));
end
title(tl, "Q1: scale-normalized LoG pyramid of flarsheim-1.jpg");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q1_log_pyramid.png"), Resolution=150);

save hw2_q1_results.mat scales stats rawMean elapsed
