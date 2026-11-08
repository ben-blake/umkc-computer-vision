%hw4_q3_retrieval - Retrieve my faces from the first 400 faces of the data set
%   Builds a database of the first cfg.nRetrieve faces of the data set and
%   cfg.nSelfDb faces of me, queries it with my remaining faces in the
%   Eigenface, Fisherface and Laplacianface spaces of Q1, and reports the
%   top cfg.topK face IDs of every query, its average precision over
%   those cfg.topK, and the mean over the queries. Also reports recall
%   at cfg.topK and the AP over the whole ranking.
%
%   AP at K is reported two ways. The first follows the example in the
%   assignment: the mean of the precision at each rank within the top K
%   that holds one of my faces, so a single hit at rank 2 is 1/2. The
%   second follows the course slide, which counts every relevant face
%   that was not retrieved as a precision of 0, so the sum is divided by
%   min(K, number of my faces in the database) instead. A query with no
%   hit scores 0 either way.
%
%   See also getAveragePrecision, hw4_q1_models, hw4_q2_recognition

%   Homework 4, Q3.

clear; clc;
cfg = hw4_config();
load(cfg.facesFile, "faces", "ids", "selfFaces", "selfIds", "selfNames");
M = load("hw4_models.mat");

dbFaces = [faces(1:cfg.nRetrieve, :); selfFaces(1:cfg.nSelfDb, :)];
dbIds = [ids(1:cfg.nRetrieve); selfIds(1:cfg.nSelfDb)];
qFaces = selfFaces(cfg.nSelfDb + 1:end, :);
qNames = selfNames(cfg.nSelfDb + 1:end);
nQ = size(qFaces, 1);
nRel = sum(dbIds == cfg.selfId);
fprintf("database: %d faces, %d of them mine; %d queries\n", numel(dbIds), nRel, nQ);

names = ["eigenface", "fisherface", "laplacianface"];
proj = {M.A0, M.A0 * M.A1, M.A0 * M.A2};
K = cfg.topK;
retrieved = zeros(nQ, K, 3);          % face IDs, query by rank by model
apK = zeros(nQ, 3);
apKSlide = zeros(nQ, 3);
recallK = zeros(nQ, 3);
apFull = zeros(nQ, 3);
for m = 1:3
    dist = pdist2(qFaces * proj{m}, dbFaces * proj{m});
    [~, order] = sort(dist, 2);
    for q = 1:nQ
        hit = dbIds(order(q, :)) == cfg.selfId;
        retrieved(q, :, m) = dbIds(order(q, 1:K));
        apK(q, m) = getAveragePrecision(hit(1:K));
        apKSlide(q, m) = getAveragePrecision(hit(1:K), NumRelevant=min(K, nRel));
        recallK(q, m) = sum(hit(1:K)) / nRel;
        apFull(q, m) = getAveragePrecision(hit);
    end
    fprintf("\n%s: top-%d face IDs, one row per query\n", names(m), K);
    for q = 1:nQ
        fprintf("  query %s, ID %d: [%s]  AP@%d = %.3f\n", qNames(q), cfg.selfId, ...
            strjoin(string(retrieved(q, :, m)), ", "), K, apK(q, m));
    end
    fprintf("  mAP@%d = %.3f (slide definition %.3f), mean recall@%d = %.3f, " + ...
        "mAP over the full ranking = %.3f\n", K, mean(apK(:, m)), mean(apKSlide(:, m)), ...
        K, mean(recallK(:, m)), mean(apFull(:, m)));
end
mapK = mean(apK, 1);
mapKSlide = mean(apKSlide, 1);
mapFull = mean(apFull, 1);

% Where my 10 database faces rank, on average, among the 410. A random
% ranking would put them at 205.5.
meanRank = zeros(1, 3);
for m = 1:3
    meanRank(m) = iMeanRank(qFaces * proj{m}, dbFaces * proj{m}, dbIds == cfg.selfId);
end
fprintf("\nmean rank of my database faces among %d (random: %.1f): %s\n", ...
    numel(dbIds), (numel(dbIds) + 1) / 2, mat2str(meanRank, 4));

% Why so low. In raw pixels, how far apart are two faces of one person,
% against two faces of different people? The data set's subjects were
% each photographed in one session; my photos span years and lighting.
Dme = pdist2(selfFaces, selfFaces);
D400 = pdist2(faces(1:cfg.nRetrieve, :), faces(1:cfg.nRetrieve, :));
same = ids(1:cfg.nRetrieve) == ids(1:cfg.nRetrieve)';
offDiag = ~eye(cfg.nRetrieve);
spread = [mean(Dme(~eye(size(Dme, 1)))), mean(D400(same & offDiag)), mean(D400(~same))];
fprintf("mean pixel distance: between two of my faces %.2f, two faces of one data set " + ...
    "subject %.2f, two different data set subjects %.2f\n", spread);

% The same retrieval with every face standardized to zero mean and unit
% standard deviation first, database and queries alike, which removes
% overall brightness and contrast.
standardize = @(F) (F - mean(F, 2)) ./ std(F, 0, 2);
mapStd = zeros(1, 3);
mapStdSlide = zeros(1, 3);
meanRankStd = zeros(1, 3);
for m = 1:3
    dist = pdist2(standardize(qFaces) * proj{m}, standardize(dbFaces) * proj{m});
    [~, order] = sort(dist, 2);
    ap = arrayfun(@(q) getAveragePrecision(dbIds(order(q, 1:K)) == cfg.selfId), 1:nQ);
    mapStd(m) = mean(ap);
    mapStdSlide(m) = mean(arrayfun(@(q) getAveragePrecision(dbIds(order(q, 1:K)) == cfg.selfId, ...
        NumRelevant=min(K, nRel)), 1:nQ));
    meanRankStd(m) = iMeanRank(standardize(qFaces) * proj{m}, ...
        standardize(dbFaces) * proj{m}, dbIds == cfg.selfId);
end
fprintf("standardized faces: mAP@%d %s (slide definition %s), mean rank %s\n", K, ...
    mat2str(mapStd, 3), mat2str(mapStdSlide, 3), mat2str(meanRankStd, 4));

% Query and its top K for every model. Green frame: one of my faces.
fig = figure(Name="Retrieval", Position=[60 60 1500 560]);
tl = tiledlayout(fig, 1, 3, TileSpacing="compact", Padding="compact");
for m = 1:3
    dist = pdist2(qFaces * proj{m}, dbFaces * proj{m});
    [~, order] = sort(dist, 2);
    canvas = ones(nQ * 24, (K + 1) * 24 + 6, 3);
    for q = 1:nQ
        r = (q - 1) * 24 + 3;
        canvas(r:r + 19, 3:22, :) = repmat(reshape(qFaces(q, :), 20, 20), 1, 1, 3);
        for k = 1:K
            j = order(q, k);
            c = k * 24 + 9;
            frame = [0.85 0.1 0.1];
            if dbIds(j) == cfg.selfId, frame = [0.1 0.7 0.1]; end
            canvas(r - 2:r + 21, c - 2:c + 21, :) = repmat(reshape(frame, 1, 1, 3), 24, 24);
            canvas(r:r + 19, c:c + 19, :) = repmat(reshape(dbFaces(j, :), 20, 20), 1, 1, 3);
        end
    end
    ax = nexttile(tl);
    image(ax, canvas); axis(ax, "image", "off");
    title(ax, sprintf("%s: mAP@%d = %.2f", names(m), K, mapK(m)));
end
title(tl, sprintf("Q3: my 5 queries (left column) and their top %d, green = my face", K));
hw4_save_figure(fig, "q3_retrieval.png");

save hw4_q3_results.mat retrieved apK apKSlide recallK apFull mapK mapKSlide mapFull qNames nRel ...
    meanRank spread mapStd mapStdSlide meanRankStd

function r = iMeanRank(Q, D, relevant)
%iMeanRank - Mean rank of the relevant database rows over all queries
[~, order] = sort(pdist2(Q, D), 2);
[~, rankOf] = sort(order, 2);
r = mean(rankOf(:, relevant), "all");
end
