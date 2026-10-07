%hw2_q2_extrema - Show the cubic fit that locates a scale extremum
%   Takes four pixels of flarsheim-1.jpg, two that are key points and two
%   that are rejected, and plots their LoG response across scale with the
%   fitted cubic, its gradient, and the extremum that
%   getScaleResponseExtrema returns. Then measures how well a cubic
%   describes the response over every candidate pixel in the image.
%
%   See also getScaleResponseExtrema, getLoGPyramid, hw2_q3_keypoints

%   Homework 2, Q2 - figures.

clear; clc;
cfg = hw2_config();
scales = cfg.scales;
N = numel(scales);

im = imread(cfg.images(1));
P = abs(getLoGPyramid(im, scales));
[H, W, ~] = size(P);
[peakOverScale, level] = max(P, [], 3);

% Two key points, one fine and one coarse, picked as the strongest whose
% fitted scale falls in each range.
[x, y, scale] = getLoGKeyPoints(im, cfg.logPeakThres);
iFine = find(scale > 1.5 & scale < 2.0, 1);
iCoarse = find(scale > 3.2 & scale < 4.2, 1);

% Two rejected pixels: the strongest whose response is largest at the
% first level, and the strongest whose response is largest at the last,
% among those for which the fit finds no extremum.
interior = false(H, W);
interior(2:end-1, 2:end-1) = true;
edgeLevels = [1 N];
rejected = zeros(2, 2);
for j = 1:2
    idx = find(level == edgeLevels(j) & interior);
    [~, order] = sort(peakOverScale(idx), "descend");
    for i = idx(order)'
        [r, c] = ind2sub([H W], i);
        if isnan(getScaleResponseExtrema(scales, reshape(P(r, c, :), 1, N)))
            rejected(j, :) = [r c];
            break
        end
    end
end
rFalling = rejected(1, 1); cFalling = rejected(1, 2);
rRising = rejected(2, 1); cRising = rejected(2, 2);

rows = [y(iFine), y(iCoarse), rFalling, rRising];
cols = [x(iFine), x(iCoarse), cFalling, cRising];
names = ["key point, fine scale", "key point, coarse scale", ...
    "rejected, falling", "rejected, rising"];

fig = figure(Name="Scale response", Position=[100 100 1100 720]);
tl = tiledlayout(fig, 2, 2, TileSpacing="compact", Padding="compact");
sFine = linspace(scales(1), scales(end), 400);
fprintf("%-26s %5s %5s %8s %8s %9s %9s\n", ...
    "pixel", "x", "y", "x0", "y_max", "f'(x0)", "f''(x0)");
examples = struct("name", {}, "x", {}, "y", {}, "response", {}, ...
    "p", {}, "x0", {}, "y_max", {});
for j = 1:4
    response = reshape(P(rows(j), cols(j), :), 1, N);
    [x0, y_max] = getScaleResponseExtrema(scales, response);
    p = polyfit(scales, response, 3);
    dp = polyder(p);

    ax = nexttile(tl);
    yyaxis(ax, "left");
    plot(ax, scales, response, "o", MarkerFaceColor="auto", MarkerSize=6);
    hold(ax, "on");
    plot(ax, sFine, polyval(p, sFine), "-", LineWidth=1.2);
    if ~isnan(x0)
        plot(ax, [x0 x0], [0 y_max], "--r", LineWidth=1);
        plot(ax, x0, y_max, "r*", MarkerSize=10);
    end
    ylabel(ax, "|L(\sigma; x, y)|");
    yyaxis(ax, "right");
    plot(ax, sFine, polyval(dp, sFine), ":", LineWidth=1.2);
    yline(ax, 0, "-", Color=[0.6 0.6 0.6]);
    ylabel(ax, "\partial L / \partial\sigma");
    xlabel(ax, "\sigma"); grid(ax, "on");
    if isnan(x0)
        title(ax, sprintf("%s at (%d, %d): no extremum", ...
            names(j), cols(j), rows(j)));
        fprintf("%-26s %5d %5d %8s %8s %9s %9s\n", names(j), ...
            cols(j), rows(j), "NaN", "NaN", "-", "-");
    else
        title(ax, sprintf("%s at (%d, %d): \\sigma_0 = %.2f, peak %.1f", ...
            names(j), cols(j), rows(j), x0, y_max));
        fprintf("%-26s %5d %5d %8.3f %8.2f %9.1e %9.2f\n", names(j), ...
            cols(j), rows(j), x0, y_max, polyval(dp, x0), ...
            polyval(polyder(dp), x0));
    end
    examples(j) = struct("name", names(j), "x", cols(j), "y", rows(j), ...
        "response", response, "p", p, "x0", x0, "y_max", y_max);
end
title(tl, "Q2: LoG response across scale, cubic fit and its gradient");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q2_scale_response.png"), Resolution=150);

% How good is a cubic as a model of the response? Fit every pixel that is
% above the threshold and a 26-neighbor maximum, which is the set the
% detector fits.
isBlockMax = imregionalmax(P, 26);
[rowGrid, colGrid] = ndgrid(1:H, 1:W);
isCandidate = isBlockMax(sub2ind([H W N], rowGrid, colGrid, level)) ...
    & (peakOverScale >= cfg.logPeakThres) & interior;
[row, col] = find(isCandidate);
nCandidate = numel(row);
r2 = zeros(nCandidate, 1);
x0All = nan(nCandidate, 1);
yMaxAll = nan(nCandidate, 1);
for i = 1:nCandidate
    response = reshape(P(row(i), col(i), :), 1, N);
    p = polyfit(scales, response, 3);
    resid = response - polyval(p, scales);
    r2(i) = 1 - sum(resid.^2) / sum((response - mean(response)).^2);
    [x0All(i), yMaxAll(i)] = getScaleResponseExtrema(scales, response);
end
candLevel = level(isCandidate);
candPeak = peakOverScale(isCandidate);
hasPeak = ~isnan(x0All);

fprintf("\ncandidates %d, with an extremum %d (%.1f%%)\n", ...
    nCandidate, sum(hasPeak), 100*mean(hasPeak));
fprintf("R^2 of the cubic: median %.3f, 10th percentile %.3f\n", ...
    median(r2), prctile(r2, 10));
fprintf("fitted peak over largest sample: median %.3f, 1st to 99th percentile %.3f to %.3f\n", ...
    median(yMaxAll(hasPeak) ./ candPeak(hasPeak)), ...
    prctile(yMaxAll(hasPeak) ./ candPeak(hasPeak), [1 99]));

fprintf("\n%5s %7s %10s %9s %10s\n", ...
    "level", "sigma", "candidates", "accepted", "median x0");
byLevel = zeros(N, 3);
for k = 1:N
    m = candLevel == k;
    byLevel(k, :) = [sum(m), sum(m & hasPeak), median(x0All(m & hasPeak))];
    fprintf("%5d %7.3f %10d %9d %10.3f\n", k, scales(k), byLevel(k, :));
end

% The fitted scale should land on or next to the level of the largest
% sample. Count how many levels apart they are.
[~, nearest] = min(abs(log(x0All(hasPeak)) - log(scales)), [], 2);
offset = histcounts(nearest - candLevel(hasPeak), -6.5:6.5);
fprintf("\nfitted level minus sampled level, -6 to +6: %s\n", mat2str(offset));

lowerScale = scales(max(candLevel - 1, 1))';
upperScale = scales(min(candLevel + 1, N))';
isNearLevel = hasPeak & (x0All >= lowerScale) & (x0All <= upperScale);
fprintf("extremum within one level of the sampled maximum: %d, further away: %d\n", ...
    sum(isNearLevel), sum(hasPeak & ~isNearLevel));
fprintf("of those, at or above the threshold (the key points): %d\n", ...
    sum(isNearLevel & yMaxAll >= cfg.logPeakThres));

fit = struct("nCandidate", nCandidate, "nWithPeak", sum(hasPeak), ...
    "r2Median", median(r2), "r2P10", prctile(r2, 10), ...
    "ratioMedian", median(yMaxAll(hasPeak) ./ candPeak(hasPeak)), ...
    "byLevel", byLevel, "levelOffset", offset, ...
    "nNearLevel", sum(isNearLevel), "nFarLevel", sum(hasPeak & ~isNearLevel), ...
    "nKeyPoint", sum(isNearLevel & yMaxAll >= cfg.logPeakThres));
save hw2_q2_results.mat examples fit
