%hw1_part2_homography - Homography estimation by SVD for two image pairs
%   Detects SIFT keypoints in each pair and matches them, takes M = 8
%   matches, builds the 16-by-9 matrix A and solves A*h = 0 with the SVD.
%   Answers questions A through D for two ways of picking the 8 points:
%   the 8 with the smallest descriptor distance, which is what the
%   assignment asks for, and 8 that are spread across the image and agree
%   with each other geometrically. Warps the first image of each pair with
%   imtransform to check the result.
%
%   See also getHomographySVD, detectSIFTFeatures, matchFeatures, imtransform

%   Homework 1, Part 2.

clear; clc; close all force;
% RANSAC is randomized and it decides which matches count as consistent,
% so the seed is fixed to keep every number in the report reproducible.
rng(0);
warning("off", "images:geotrans:transformationMatrixBadlyConditioned");
pairs = ["sift-1.jpg" "sift-2.jpg"; "sift-3.png" "sift-4.png"];
M = 8;              % correspondences used to build A
tau = 3;            % agreement threshold in pixels
sweep = [1 3 5 10];
res = struct([]);

for pr = 1:size(pairs, 1)
    nameA = pairs(pr,1); nameB = pairs(pr,2);
    fprintf("\n============ pair %d:  %s  ->  %s ============\n", pr, nameA, nameB);
    I1 = imread(nameA); I2 = imread(nameB);
    G1 = im2gray(I1);   G2 = im2gray(I2);

    % ---------- (A) SIFT keypoints and matches ------------------------
    k1 = detectSIFTFeatures(G1);
    k2 = detectSIFTFeatures(G2);
    [f1, v1] = extractFeatures(G1, k1);
    [f2, v2] = extractFeatures(G2, k2);
    % MatchThreshold is raised from its default of 1.0 because at the
    % default these two pairs return too few matches to work with.
    [idx, scores] = matchFeatures(f1, f2, Unique=true, MaxRatio=0.7, ...
        MatchThreshold=10);
    m1 = double(v1(idx(:,1)).Location);
    m2 = double(v2(idx(:,2)).Location);
    n = size(idx, 1);
    fprintf("A) keypoints %d and %d, matched pairs %d\n", k1.Count, k2.Count, n);

    % Reference homography from all matches, used only to label which
    % matches are geometrically consistent. It plays no part in the SVD.
    tfRef = estgeotform2d(m1, m2, "projective", MaxDistance=tau, ...
        Confidence=99.9, MaxNumTrials=10000);
    Href = double(tfRef.A);
    consistent = iSymmetricError(Href, m1, m2) <= tau;
    fprintf("   geometrically consistent matches: %d of %d (%.0f%%)\n", ...
        sum(consistent), n, 100*sum(consistent)/n);

    % ---------- two ways of choosing the 8 points ---------------------
    [~, byScore] = sort(scores, "ascend");
    pick.naive = byScore(1:M)';
    minSep = 0.15 * hypot(size(I1,1), size(I1,2));
    pick.spread = iSelectSpread(m1, scores, find(consistent), M, minSep);

    for which = ["naive" "spread"]
        sel = pick.(which);
        b1 = m1(sel,:); b2 = m2(sel,:);
        if which == "naive"
            fprintf("\n--- 8 points with the smallest descriptor distance ---\n");
        else
            fprintf("\n--- 8 consistent points spread over the image ---\n");
        end
        nUnique = size(unique(b1, "rows"), 1);
        fprintf("   distinct source points: %d of %d, spread %.0f%% x %.0f%% of frame\n", ...
            nUnique, M, 100*range(b1(:,1))/size(I1,2), 100*range(b1(:,2))/size(I1,1));

        % ---------- (B) and (C) SVD of A ------------------------------
        [H, sv, h, A] = getHomographySVD(b1, b2);
        fprintf("B) size(A) = [%d %d], rank(A) = %d\n", size(A,1), size(A,2), rank(A));
        fprintf("     s = ["); fprintf(" %.4g", sv); fprintf(" ]\n");
        fprintf("     s9 = %.4g, s9/s8 = %.3e, s9/s1 = %.3e, cond(A) = %.3e\n", ...
            sv(9), sv(9)/sv(8), sv(9)/sv(1), cond(A));
        fprintf("C) h' = ["); fprintf(" %+.5e", h); fprintf(" ]\n");
        fprintf("     norm(h) = %.6f, norm(A*h) = %.4e\n", norm(h), norm(A*h));
        fprintf("     H =\n"); disp(H);
        e8 = iTransferError(H, b1, b2);
        fprintf("     max reprojection error on the 8 points: %.3f px\n", max(e8));

        % ---------- (D) agreement over all matches --------------------
        err = iSymmetricError(H, m1, m2);
        cnt = arrayfun(@(t) sum(err <= t), sweep);
        fprintf("D) agree at %g px: %d of %d (%.1f%%), median error %.2f px\n", ...
            tau, sum(err <= tau), n, 100*sum(err <= tau)/n, median(err));
        fprintf("     sweep"); fprintf("  %gpx:%d", [sweep; cnt]); fprintf("\n");

        r = struct("pair", pr, "which", which, "nKey", [k1.Count k2.Count], ...
            "nMatch", n, "nConsistent", sum(consistent), "sel", sel, ...
            "sv", sv, "h", h, "H", H, "A", A, "rankA", rank(A), ...
            "condA", cond(A), "maxErr8", max(e8), "err", err, ...
            "agree", sum(err <= tau), "sweep", cnt, "b1", b1, "b2", b2);
        if isempty(res), res = r; else, res(end+1) = r; end %#ok<SAGROW>
    end

    % ---------- figures ------------------------------------------------
    figM = figure(Name=sprintf("pair%d matches", pr), Position=[60 60 900 980], ...
        Visible="off");
    tl = tiledlayout(figM, 3, 1, TileSpacing="compact", Padding="compact");
    ax = nexttile(tl,1);
    showMatchedFeatures(I1, I2, m1, m2, "montage", Parent=ax);
    title(ax, sprintf("all %d SIFT matches", n));
    ax = nexttile(tl,2);
    showMatchedFeatures(I1, I2, m1(pick.naive,:), m2(pick.naive,:), ...
        "montage", Parent=ax);
    title(ax, "the 8 smallest descriptor distances, clustered");
    ax = nexttile(tl,3);
    showMatchedFeatures(I1, I2, m1(pick.spread,:), m2(pick.spread,:), ...
        "montage", Parent=ax);
    title(ax, "the 8 spread points used for the final H");
    theme(figM, "light"); drawnow;
    exportgraphics(figM, sprintf("results/part2_pair%d_matches.png", pr), Resolution=150);

    % imtransform works on row vectors, so it takes the transpose of H.
    Hs = res(end).H;
    T = maketform("projective", Hs');
    W = imtransform(I1, T, XData=[1 size(I2,2)], YData=[1 size(I2,1)], FillValues=0);
    figW = figure(Name=sprintf("pair%d warp", pr), Position=[60 60 1150 420], ...
        Visible="off");
    tl = tiledlayout(figW, 1, 3, TileSpacing="compact", Padding="compact");
    ax = nexttile(tl,1); imshow(W, Parent=ax);       title(ax, "image 1 warped by H");
    ax = nexttile(tl,2); imshow(I2, Parent=ax);      title(ax, "image 2, the target");
    ax = nexttile(tl,3); imshowpair(W, I2, "falsecolor", Parent=ax);
    title(ax, "overlay of the two");
    theme(figW, "light"); drawnow;
    exportgraphics(figW, sprintf("results/part2_pair%d_warp.png", pr), Resolution=150);
end
save("hw1_part2_results.mat", "res", "M", "tau", "sweep", "pairs");

function sel = iSelectSpread(p, scores, pool, M, minSep)
%iSelectSpread - Best-scoring points that are at least minSep apart
%   Walks the candidate pool in order of descriptor distance and keeps a
%   point only if it is far enough from every point already kept, so the
%   returned set covers the frame instead of clustering in one corner.
[~, ord] = sort(scores(pool), "ascend");
pool = pool(ord);
sel = [];
for i = pool(:)'
    if isempty(sel) || min(vecnorm(p(sel,:) - p(i,:), 2, 2)) >= minSep
        sel(end+1) = i; %#ok<AGROW>
    end
    if numel(sel) == M, break; end
end
if numel(sel) < M                    % relax if the frame is too small
    extra = setdiff(pool(:)', sel, "stable");
    sel = [sel, extra(1:M-numel(sel))];
end
end

function e = iTransferError(H, p1, p2)
%iTransferError - Forward reprojection error in pixels
q = (H * [p1, ones(size(p1,1),1)]')';
e = vecnorm(q(:,1:2)./q(:,3) - p2, 2, 2);
end

function e = iSymmetricError(H, p1, p2)
%iSymmetricError - Mean of the forward and backward transfer errors
e = 0.5 * (iTransferError(H, p1, p2) + iTransferError(inv(H), p2, p1));
end
