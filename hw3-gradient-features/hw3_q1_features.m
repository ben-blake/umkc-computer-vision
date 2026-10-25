%hw3_q1_features - Extract SIFT, dense SIFT and HoG features of all images
%   Runs getImageFeatures with each of the three feature types on the 600
%   images of the 15-class NWPU subset and saves the results as the cell
%   arrays f_sift, f_dsift and f_hog in gradient_features.mat, one cell
%   per image. Prints how many features each type gives per image and
%   draws all three on one example image.
%
%   See also getImageFeatures, hw3_config, hw3_q2_gmm

%   Homework 3, Q1.

clear; clc;
cfg = hw3_config();
N = numel(cfg.files);
nType = numel(cfg.types);
matFile = "gradient_features.mat";

if isfile(matFile)
    % About a minute for all three. Delete the MAT file to recompute.
    fprintf("Loading features from %s\n", matFile);
    load(matFile, "f_sift", "f_dsift", "f_hog", "seconds");
else
    feats = cell(N, nType);
    seconds = zeros(1, nType);
    for t = 1:nType
        opt = struct("type", cfg.types(t));
        tStart = tic;
        for i = 1:N
            feats{i, t} = getImageFeatures(imread(cfg.files(i)), opt);
        end
        seconds(t) = toc(tStart);
        fprintf("%-10s %6.1f s\n", cfg.typeNames(t), seconds(t));
    end
    % SIFT and dense SIFT entries are whole numbers from 0 to 255, so
    % uint8 stores them exactly in a quarter of the space.
    f_sift = cellfun(@uint8, feats(:, 1), UniformOutput=false);
    f_dsift = cellfun(@uint8, feats(:, 2), UniformOutput=false);
    f_hog = feats(:, 3);
    labels = cfg.labels;
    classes = cfg.classes;
    save(matFile, "f_sift", "f_dsift", "f_hog", "labels", "classes", "seconds");
end
allFeats = {f_sift, f_dsift, f_hog};

% Features per image. SIFT depends on the content, the other two only on
% the image size.
counts = zeros(N, nType);
for t = 1:nType
    counts(:, t) = cellfun(@(f) size(f, 1), allFeats{t});
end
fprintf("\n%-11s %4s %7s %7s %7s %7s %10s %7s\n", "feature", "d", "min", ...
    "median", "mean", "max", "total", "s/img");
for t = 1:nType
    fprintf("%-11s %4d %7d %7d %7.1f %7d %10d %7.3f\n", cfg.typeNames(t), ...
        size(allFeats{t}{1}, 2), min(counts(:, t)), median(counts(:, t)), ...
        mean(counts(:, t)), max(counts(:, t)), sum(counts(:, t)), seconds(t)/N);
end

nClass = numel(cfg.classes);
siftByClass = zeros(nClass, 3);
for k = 1:nClass
    c = counts(cfg.labels == k, 1);
    siftByClass(k, :) = [min(c), median(c), max(c)];
end
fprintf("\nSIFT key points per image, by class\n");
disp(array2table(siftByClass, RowNames=cfg.classes, ...
    VariableNames=["min" "median" "max"]));

% One image with all three features on it.
idx = find(cfg.labels == 1, 1);
im = imread(cfg.files(idx));
gray = im2gray(im);
points = detectSIFTFeatures(gray);
[~, locD] = getImageFeatures(im, struct("type", "dsft"));
[~, hogVis] = extractHOGFeatures(gray, CellSize=[8 8]);

fig = figure(Name="Gradient features", Position=[100 100 1300 380]);
tl = tiledlayout(fig, 1, 4, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
imshow(gray, Parent=ax); hold(ax, "on");
plot(points.selectStrongest(150), ax, ShowScale=true, ShowOrientation=true);
title(ax, sprintf("SIFT: strongest 150 of %d", points.Count));
ax = nexttile(tl);
imshow(gray, Parent=ax); hold(ax, "on");
plot(ax, locD(:, 1), locD(:, 2), "y.", MarkerSize=6);
rectangle(ax, Position=[locD(1, :) - 16, 32, 32], EdgeColor="r", LineWidth=1.2);
title(ax, sprintf("dense SIFT: %d points, step 8, 32 px window", size(locD, 1)));
ax = nexttile(tl);
imshow(gray, Parent=ax); hold(ax, "on");
plot(hogVis, ax, Color="y");
title(ax, sprintf("HoG: %d blocks of 2 by 2 cells, 8 px", size(f_hog{idx}, 1)));
ax = nexttile(tl);
boxchart(ax, categorical(cfg.labels, 1:nClass, strrep(cfg.classes, "_", " ")), ...
    counts(:, 1));
ylabel(ax, "SIFT key points per image"); grid(ax, "on");
title(ax, "SIFT key points by class");
title(tl, "Q1: the three gradient features on " + cfg.classes(1) + " image 1", ...
    Interpreter="none");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q1_features.png"), Resolution=150);

save hw3_q1_results.mat counts siftByClass seconds
