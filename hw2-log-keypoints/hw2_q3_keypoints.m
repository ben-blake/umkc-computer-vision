%hw2_q3_keypoints - LoG key points on every test image, compared to SIFT
%   Runs getLoGKeyPoints and detectSIFTFeatures on the three test images,
%   draws the two detections side by side, and measures how far they
%   agree: how many key points each finds, at which scales, how many LoG
%   key points have a SIFT key point at the same place, and how many of
%   them SIFT would have thrown out as edge responses.
%
%   See also getLoGKeyPoints, detectSIFTFeatures, hw2_config, hw2_q3_export

%   Homework 2, Q3 - detection and comparison.

clear; clc;
cfg = hw2_config();
scales = cfg.scales;
N = numel(scales);
nImage = numel(cfg.images);
nShow = cfg.budgets(1);

res = struct();
for n = 1:nImage
    [~, name, ext] = fileparts(cfg.images(n));
    im = imread(cfg.images(n));
    gray = im2gray(im);
    [H, W] = size(gray);

    tStart = tic;
    [x, y, scale, strength] = getLoGKeyPoints(im, cfg.logPeakThres);
    tLoG = toc(tStart);

    tStart = tic;
    sift = detectSIFTFeatures(gray);
    tSift = toc(tStart);
    % detectSIFTFeatures returns points in no particular order.
    [~, order] = sort(sift.Metric, "descend");
    sift = sift(order);
    siftXY = double(sift.Location);
    siftScale = double(sift.Scale);
    % SIFT searches many octaves, the LoG pyramid only one range of
    % scales. Only the SIFT points inside that range are comparable.
    inRange = siftScale >= scales(1) & siftScale <= scales(end);

    r = struct();
    r.image = name + ext;
    r.size = [H W];
    r.nLoG = numel(x);
    r.nSift = sift.Count;
    r.nSiftInRange = sum(inRange);
    r.tLoG = tLoG;
    r.tSift = tSift;
    r.logXY = [x y];
    r.logScale = scale;
    r.logStrength = strength;
    r.siftXY = siftXY;
    r.siftScale = siftScale;
    r.siftMetric = double(sift.Metric);

    % Count versus threshold.
    r.sweep = zeros(size(cfg.thresSweep));
    for j = 1:numel(cfg.thresSweep)
        r.sweep(j) = numel(getLoGKeyPoints(im, cfg.thresSweep(j)));
    end

    % Agreement in position. A LoG key point agrees when a SIFT key point
    % lies within matchRadius pixels. The chance figure repeats the count
    % against the SIFT points mirrored left to right, which keeps their
    % number and their clustering but breaks any real correspondence.
    siftIn = siftXY(inRange, :);
    siftInScale = siftScale(inRange);
    mirrored = [W + 1 - siftIn(:, 1), siftIn(:, 2)];
    [dist, nearest] = min(pdist2([x y], siftIn), [], 2);
    distChance = min(pdist2([x y], mirrored), [], 2);
    isMatch = dist <= cfg.matchRadius;
    r.matchAll = [sum(isMatch), numel(x), sum(distChance <= cfg.matchRadius)];
    r.medianDist = median(dist);

    % The other direction: SIFT key points that have a LoG key point.
    distBack = min(pdist2(siftIn, [x y]), [], 2);
    r.matchBack = [sum(distBack <= cfg.matchRadius), size(siftIn, 1)];

    % Agreement in scale, over the matched pairs only.
    ratio = scale(isMatch) ./ siftInScale(nearest(isMatch));
    r.scaleRatio = [median(ratio), prctile(ratio, 25), prctile(ratio, 75)];
    r.scaleCorr = corr(log(scale(isMatch)), ...
        log(siftInScale(nearest(isMatch))), Type="Spearman");

    % Same question at an equal budget: the strongest K of each.
    r.matchBudget = zeros(numel(cfg.budgets), 3);
    for j = 1:numel(cfg.budgets)
        k = min([cfg.budgets(j), numel(x), size(siftIn, 1)]);
        d = min(pdist2([x(1:k) y(1:k)], siftIn(1:k, :)), [], 2);
        dc = min(pdist2([x(1:k) y(1:k)], mirrored(1:k, :)), [], 2);
        r.matchBudget(j, :) = [sum(d <= cfg.matchRadius), k, ...
            sum(dc <= cfg.matchRadius)];
    end

    % Edge test. SIFT drops a key point when the two principal curvatures
    % of its response differ by more than edgeRatio. The LoG detector has
    % no such test, so count how many of its key points would fail.
    P = abs(getLoGPyramid(im, scales));
    [~, lev] = min(abs(log(scale) - log(scales)), [], 2);
    isEdge = false(numel(x), 1);
    for i = 1:numel(x)
        D = P(y(i)-1:y(i)+1, x(i)-1:x(i)+1, lev(i));
        dxx = D(2,3) - 2*D(2,2) + D(2,1);
        dyy = D(3,2) - 2*D(2,2) + D(1,2);
        dxy = (D(3,3) - D(3,1) - D(1,3) + D(1,1)) / 4;
        tr = dxx + dyy;
        dt = dxx*dyy - dxy^2;
        isEdge(i) = dt <= 0 || tr^2/dt >= (cfg.edgeRatio + 1)^2/cfg.edgeRatio;
    end
    r.nEdge = sum(isEdge);
    r.isEdge = isEdge;
    r.matchEdge = [sum(isMatch & isEdge), sum(isEdge)];
    r.matchNotEdge = [sum(isMatch & ~isEdge), sum(~isEdge)];

    % Histogram of scales on the pyramid levels.
    edges = sqrt([scales(1)^3/scales(2), scales(1:end-1).*scales(2:end), ...
        scales(end)^3/scales(end-1)]);
    r.logScaleHist = histcounts(scale, edges);
    r.siftScaleHist = histcounts(siftInScale, edges);

    if n == 1
        res = r;
    else
        res(n) = r;
    end

    % Side by side: strongest nShow of each, circle radius sqrt(2)*sigma.
    fig = figure(Name="Key points " + name, Position=[100 100 1300 520]);
    tl = tiledlayout(fig, 1, 2, TileSpacing="compact", Padding="compact");
    k = min(nShow, sift.Count);
    ax = nexttile(tl);
    imshow(gray, Parent=ax); hold(ax, "on");
    viscircles(ax, siftXY(1:k, :), sqrt(2)*siftScale(1:k), ...
        Color="y", LineWidth=0.6, EnhanceVisibility=false);
    plot(ax, siftXY(1:k, 1), siftXY(1:k, 2), "y+", MarkerSize=5);
    title(ax, sprintf("SIFT: strongest %d of %d", k, sift.Count));
    k = min(nShow, numel(x));
    ax = nexttile(tl);
    imshow(gray, Parent=ax); hold(ax, "on");
    viscircles(ax, [x(1:k) y(1:k)], sqrt(2)*scale(1:k), ...
        Color="r", LineWidth=0.6, EnhanceVisibility=false);
    plot(ax, x(1:k), y(1:k), "r+", MarkerSize=5);
    title(ax, sprintf("LoG: strongest %d of %d, threshold %g", ...
        k, numel(x), cfg.logPeakThres));
    title(tl, "Q3: " + name + ext, Interpreter="none");
    theme(fig, "light");
    drawnow;
    exportgraphics(fig, fullfile(cfg.resultsDir, ...
        sprintf("q3_keypoints_%d.png", n)), Resolution=150);
end

% The handout's view for the first image: detections on the image, and
% on the maximum of the LoG magnitude over scale.
im = imread(cfg.images(1));
gray = im2gray(im);
logMax = max(abs(getLoGPyramid(im, scales)), [], 3);
k = min(nShow, res(1).nLoG);
fig = figure(Name="LoG max", Position=[100 100 1300 520]);
tl = tiledlayout(fig, 1, 2, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
imshow(gray, Parent=ax); hold(ax, "on");
plot(ax, res(1).logXY(1:k, 1), res(1).logXY(1:k, 2), "r+", MarkerSize=6);
title(ax, sprintf("strongest %d LoG key points", k));
ax = nexttile(tl);
imagesc(ax, logMax, [0 prctile(logMax(:), 99.5)]);
axis(ax, "image"); axis(ax, "off"); colormap(ax, "gray"); hold(ax, "on");
plot(ax, res(1).logXY(1:k, 1), res(1).logXY(1:k, 2), "r+", MarkerSize=6);
title(ax, "log max: largest |LoG| over the seven scales");
title(tl, "Q3: LoG key points on flarsheim-1.jpg");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q3_log_max.png"), Resolution=150);

% Scale distributions and count versus threshold, all images.
fig = figure(Name="Scales and thresholds", Position=[100 100 1300 420]);
tl = tiledlayout(fig, 1, nImage + 1, TileSpacing="compact", Padding="compact");
for n = 1:nImage
    ax = nexttile(tl);
    share = 100 * [res(n).siftScaleHist / sum(res(n).siftScaleHist); ...
        res(n).logScaleHist / sum(res(n).logScaleHist)]';
    bar(ax, share);
    xticks(ax, 1:N); xticklabels(ax, compose("%.2f", scales));
    xlabel(ax, "\sigma of nearest pyramid level");
    ylabel(ax, "% of key points");
    legend(ax, ["SIFT, in range", "LoG"], Location="northeast");
    title(ax, res(n).image, Interpreter="none");
end
ax = nexttile(tl);
sweep = vertcat(res.sweep)';
semilogy(ax, cfg.thresSweep, sweep, "-o", LineWidth=1.2);
hold(ax, "on"); grid(ax, "on");
xlabel(ax, "log\_peak\_thres"); ylabel(ax, "LoG key points");
legend(ax, [res.image], Interpreter="none", Location="southwest");
title(ax, "count versus threshold");
title(tl, "Q3: where the key points sit in scale, and how many there are");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q3_scales_thresholds.png"), Resolution=150);

% Tables.
fprintf("%-16s %6s %6s %9s %7s %7s\n", ...
    "image", "LoG", "SIFT", "in range", "t LoG", "t SIFT");
for n = 1:nImage
    fprintf("%-16s %6d %6d %9d %6.2fs %6.2fs\n", res(n).image, ...
        res(n).nLoG, res(n).nSift, res(n).nSiftInRange, ...
        res(n).tLoG, res(n).tSift);
end

fprintf("\nLoG key points by threshold\n%-16s", "image");
fprintf(" %6d", cfg.thresSweep); fprintf("\n");
for n = 1:nImage
    fprintf("%-16s", res(n).image); fprintf(" %6d", res(n).sweep); fprintf("\n");
end

fprintf("\nLoG key points with a SIFT key point within %g px\n", cfg.matchRadius);
fprintf("%-16s %16s %9s %16s %10s\n", ...
    "image", "matched", "chance", "SIFT with LoG", "median px");
for n = 1:nImage
    m = res(n).matchAll; b = res(n).matchBack;
    fprintf("%-16s %5d/%5d %4.1f%% %8.1f%% %5d/%5d %4.1f%% %10.2f\n", ...
        res(n).image, m(1), m(2), 100*m(1)/m(2), 100*m(3)/m(2), ...
        b(1), b(2), 100*b(1)/b(2), res(n).medianDist);
end

fprintf("\nSame, strongest K of each detector (matched%%, chance%%)\n");
for n = 1:nImage
    fprintf("%-16s", res(n).image);
    for j = 1:numel(cfg.budgets)
        m = res(n).matchBudget(j, :);
        fprintf("  K=%4d: %5.1f%% (%4.1f%%)", m(2), 100*m(1)/m(2), 100*m(3)/m(2));
    end
    fprintf("\n");
end

fprintf("\nScale of matched pairs, LoG sigma over SIFT sigma\n");
fprintf("%-16s %8s %8s %8s %10s\n", "image", "median", "q25", "q75", "Spearman");
for n = 1:nImage
    fprintf("%-16s %8.3f %8.3f %8.3f %10.3f\n", res(n).image, ...
        res(n).scaleRatio, res(n).scaleCorr);
end

fprintf("\nLoG key points that fail the SIFT edge test, r = %g\n", cfg.edgeRatio);
fprintf("%-16s %14s %22s %22s\n", ...
    "image", "edge-like", "matched if edge-like", "matched if not");
for n = 1:nImage
    e = res(n).matchEdge; g = res(n).matchNotEdge;
    fprintf("%-16s %5d (%4.1f%%) %12d/%4d %4.1f%% %12d/%4d %4.1f%%\n", ...
        res(n).image, res(n).nEdge, 100*res(n).nEdge/res(n).nLoG, ...
        e(1), e(2), 100*e(1)/max(e(2), 1), g(1), g(2), 100*g(1)/max(g(2), 1));
end

save hw2_q3_results.mat res
