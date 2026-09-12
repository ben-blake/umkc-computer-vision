%hw1_part1_colortable - Build the HSV color code books for K = 64 and 128
%   Samples pixels from the 600 images of the 15-class NWPU subset,
%   clusters them in HSV space with k-means, and saves the code books to
%   color_table64.mat and color_table128.mat. Also draws the code books
%   as color strips and as a scatter in the hue-saturation plane.
%
%   See also getHSVColorTable, hw1_part2_hist, hw1_part3_recog

%   Homework 1, Part 1.

clear; clc;
[files, labels, classes] = hw1_config();
fprintf("Data set: %d images, %d classes, %d per class\n", ...
    numel(files), numel(classes), numel(files)/numel(classes));

KList = [64 128];
tables = cell(size(KList));
for i = 1:numel(KList)
    K = KList(i);
    matFile = sprintf("color_table%d.mat", K);
    if isfile(matFile)
        % k-means takes about a minute per code book, so reuse the
        % saved result and delete the MAT file to force a retrain.
        fprintf("Loading K = %3d code book from %s\n", K, matFile);
        load(matFile, "color_table");
    else
        fprintf("Training K = %3d code book ... ", K);
        t0 = tic;
        color_table = getHSVColorTable(files, K, PixelsPerImage=500);
        fprintf("%.1f s\n", toc(t0));
        save(matFile, "color_table");
    end
    tables{i} = color_table;
end

% Report how the code words are spread over the HSV cube.
for i = 1:numel(KList)
    ct = tables{i};
    fprintf("K = %3d: H [%.2f %.2f]  S [%.2f %.2f]  V [%.2f %.2f]\n", ...
        KList(i), min(ct(:,1)), max(ct(:,1)), min(ct(:,2)), ...
        max(ct(:,2)), min(ct(:,3)), max(ct(:,3)));
end

fig = figure(Name="HSV color tables", Position=[100 100 900 600]);
tl = tiledlayout(fig, 2, 2, TileSpacing="compact", Padding="compact");
for i = 1:numel(KList)
    ct = tables{i};
    K = KList(i);

    ax = nexttile(tl, i);
    nRow = 8;
    strip = reshape(hsv2rgb(ct), [nRow, K/nRow, 3]);
    image(ax, permute(strip, [2 1 3]));
    axis(ax, "image"); axis(ax, "off");
    title(ax, sprintf("K = %d code words", K));

    ax = nexttile(tl, i+2);
    scatter(ax, ct(:,1), ct(:,2), 40, hsv2rgb(ct), "filled", ...
        MarkerEdgeColor=[0.3 0.3 0.3]);
    xlabel(ax, "hue"); ylabel(ax, "saturation");
    xlim(ax, [0 1]); ylim(ax, [0 1]); grid(ax, "on");
    title(ax, sprintf("K = %d, hue vs saturation", K));
end
title(tl, "Part 1: HSV k-means color tables");
theme(fig, "light");
drawnow;
exportgraphics(fig, "results/part1_color_tables.png", Resolution=150);
