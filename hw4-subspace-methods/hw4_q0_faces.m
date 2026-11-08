%hw4_q0_faces - Crop my photos and the celebrity photos into the data set
%   Runs getFaceCrop on the 15 photos of me in self/ and the 10 celebrity
%   photos in celebrity/, which gives 20-by-20 faces in [0, 1] like the
%   ones in the data set. The celebrity becomes subject 418, appended to
%   the 6680 faces as the assignment asks. My faces keep my student ID
%   and are stored apart from the data set, because Q3 decides which of
%   them go into the database and which are queries. Everything is saved
%   to hw4_faces.mat, and the crops are drawn next to faces from the
%   data set.
%
%   See also getFaceCrop, hw4_config, hw4_q1_models

%   Homework 4, data preparation.

clear; clc;
cfg = hw4_config();
S = load(cfg.dataFile, "faces", "ids");

selfFiles = iImageFiles(cfg.selfDir);
celebFiles = iImageFiles(cfg.celebDir);
fprintf("%d photos of me, %d celebrity photos\n", numel(selfFiles), numel(celebFiles));

[selfFaces, selfBoxes] = iCropAll(selfFiles);
[celebFaces, celebBoxes] = iCropAll(celebFiles);
selfIds = repmat(cfg.selfId, numel(selfFiles), 1);
celebIds = repmat(cfg.celebId, numel(celebFiles), 1);

% The augmented data set: the provided faces with the celebrity appended,
% so the first 800 and first 400 faces stay the ones the assignment means.
faces = [S.faces; celebFaces];
ids = [S.ids; celebIds];
[~, selfNames] = fileparts(selfFiles);
[~, celebNames] = fileparts(celebFiles);
save(cfg.facesFile, "faces", "ids", "selfFaces", "selfIds", "selfNames", ...
    "celebFaces", "celebIds", "celebNames", "selfBoxes", "celebBoxes");
fprintf("saved %s: %d faces of %d subjects, plus %d of me\n", ...
    cfg.facesFile, size(faces, 1), numel(unique(ids)), size(selfFaces, 1));

% How close the crops are to the data set in brightness and contrast.
stats = @(F) [mean(F(:)), mean(std(F, 0, 2))];
fprintf("\n%-12s %10s %10s\n", "faces", "mean", "std/face");
fprintf("%-12s %10.3f %10.3f\n", "data set", stats(S.faces));
fprintf("%-12s %10.3f %10.3f\n", "celebrity", stats(celebFaces));
fprintf("%-12s %10.3f %10.3f\n", "me", stats(selfFaces));

% My 15, the celebrity's 10 and 10 faces from the data set.
rng(cfg.seed);
sample = sort(randperm(size(S.faces, 1), 10));
rowsOf = {selfFaces, celebFaces, S.faces(sample, :)};
titles = ["me, ID " + cfg.selfId + ": 1 to 10 database, 11 to 15 queries", ...
    "celebrity, ID " + cfg.celebId, "data set, 10 at random"];
fig = figure(Name="Face crops", Position=[100 100 1300 330]);
tl = tiledlayout(fig, 3, 1, TileSpacing="compact", Padding="compact");
for r = 1:3
    ax = nexttile(tl);
    F = rowsOf{r};
    strip = ones(20, 21 * 15 - 1);
    for k = 1:size(F, 1)
        strip(:, (k-1)*21 + (1:20)) = reshape(F(k, :), 20, 20);
    end
    imagesc(ax, strip, [0 1]); colormap(ax, "gray"); axis(ax, "image", "off");
    title(ax, titles(r));
end
title(tl, "The added faces, cropped to 20 by 20, next to the data set");
hw4_save_figure(fig, "q0_face_crops.png");

function files = iImageFiles(folder)
%iImageFiles - Image files in a folder, sorted by name
d = [dir(fullfile(folder, "*.png")); dir(fullfile(folder, "*.PNG")); ...
    dir(fullfile(folder, "*.jpg")); dir(fullfile(folder, "*.JPG")); ...
    dir(fullfile(folder, "*.jpeg"))];
files = unique(string(fullfile({d.folder}, {d.name})'));
end

function [F, boxes] = iCropAll(files)
%iCropAll - getFaceCrop on every file, one face per row
F = zeros(numel(files), 400);
boxes = zeros(numel(files), 4);
for k = 1:numel(files)
    [F(k, :), boxes(k, :)] = getFaceCrop(imread(files(k)));
end
end
