%hw3_verify - Independent checks on the Homework 3 functions and results
%   Tests getImageFeatures, getFisherVectorModel, getFisherVector, getVLAD
%   and hw3_loo against cases whose answer is known in advance or against
%   a second implementation that shares no code with them: a rotated image
%   for dense SIFT, per-point HoG, GMMs fitted to data drawn from a known
%   mixture, and Fisher and VLAD vectors written out one component at a
%   time with mvnpdf. Then confirms that the saved results reproduce.
%   Prints PASS or FAIL for each check.
%
%   See also getImageFeatures, getFisherVector, getVLAD, hw3_loo

%   Homework 3 - verification harness.

clear; clc; rng(0);
chk = @(tf, name) fprintf("%-70s %s\n", name, ...
    string(tf).replace("true","PASS").replace("false","FAIL"));
cfg = hw3_config();
im = imread(cfg.files(1));
gray = im2gray(im);

fprintf("--- data set ---\n");
chk(numel(cfg.files) == 600 && numel(unique(cfg.files)) == 600, "600 distinct images");
chk(all(isfile(cfg.files)), "every file exists on disk");
chk(isequal(cfg.labels, repelem((1:15)', 40)), "labels are 40 of each class 1..15");
hw1 = fullfile(fileparts(pwd), "hw1-image-formation", "NWPU-RESISC45");
if isfolder(hw1)
    same = true;
    for i = [1 222 600]
        [~, name, ext] = fileparts(cfg.files(i));
        other = fullfile(hw1, cfg.classes(cfg.labels(i)), name + ext);
        same = same && isequal(imread(cfg.files(i)), imread(other));
    end
    chk(same, "images 1, 222 and 600 are the ones Homework 1 used");
end

fprintf("\n--- Q1: getImageFeatures ---\n");
fS = getImageFeatures(im, struct("type", "sift"));
fD = getImageFeatures(im, struct("type", "dsft"));
fH = getImageFeatures(im, struct("type", "hogf"));
chk(size(fS, 2) == 128 && size(fD, 2) == 128 && size(fH, 2) == 36, ...
    "dimensions are 128, 128 and 36");
chk(isequal(getImageFeatures(im), fS), "the default type is sift");
chk(isequal(getImageFeatures(im, struct("type", "dsft")), ...
    getImageFeatures(gray, struct("type", "dsft"))), "RGB input equals its grayscale");

ref = extractFeatures(gray, detectSIFTFeatures(gray));
chk(isequal(fS, single(ref)), "sift is detectSIFTFeatures + extractFeatures");
chk(all(fS(:) == round(fS(:))) && all(fS(:) >= 0 & fS(:) <= 255), ...
    "sift entries are whole numbers in 0..255, so uint8 is lossless");
chk(all(abs(vecnorm(double(fD), 2, 2) - 512) < 2), ...
    "dense SIFT has norm 512 like the toolbox SIFT (to rounding)");
chk(all(fD(:) == round(fD(:))) && all(fD(:) >= 0 & fD(:) <= 255), ...
    "dense SIFT entries are whole numbers in 0..255, like the toolbox SIFT");

% A vertical step edge has a horizontal gradient only, so all of its
% energy must fall in orientation bins 1 and 5, 0 and 180 degrees.
step = [zeros(64, 32), ones(64, 32)];
[fE, locE] = getImageFeatures(step, struct("type", "dsft"));
fE = fE(abs(locE(:, 1) - 32.5) < 8, :);
byOri = sum(reshape(double(fE)', 8, []), 2);
chk(sum(byOri([1 5])) / sum(byOri) > 0.999, ...
    "step edge: dense SIFT energy only in the 0 and 180 degree bins");

% Rotating the image by 90 degrees must rotate the 4x4 spatial grid and
% shift every orientation by two bins. A 255-pixel image keeps the grid
% symmetric, so every grid point lands on a grid point.
sq = im2double(gray(1:255, 1:255));
[f0, l0] = getImageFeatures(sq, struct("type", "dsft"));
[f1, l1] = getImageFeatures(rot90(sq), struct("type", "dsft"));
% rot90 sends pixel (r, c) to (256 - c, r).
[~, j] = ismember([l0(:, 2), 256 - l0(:, 1)], l1, "rows");
D0 = reshape(double(f0)', 8, 4, 4, []);      % ori, x, y, point
D1 = reshape(double(f1(j, :))', 8, 4, 4, []);
best = inf;
for s = [2 -2]                               % either sign convention
    for dim = [2 3]                          % and either mirror
        P = circshift(D0, s, 1);
        P = flip(permute(P, [1 3 2 4]), dim); % swap the x and y bins
        best = min(best, max(abs(P(:) - D1(:))));
    end
end
chk(all(j > 0) && best <= 1, sprintf( ...
    "rotating 90 degrees permutes the dense SIFT bins (max diff %g)", best));

pts = [17 17; 57 17; 17 41; 121 201];
hp = extractHOGFeatures(gray, pts, CellSize=[8 8], BlockSize=[2 2]);
[d, ~] = min(pdist2(double(hp), double(fH)), [], 2);
chk(max(d) < 1e-6, "every HoG row is a block extractHOGFeatures gives at a point");
chk(threw(@() getImageFeatures(im, struct("type", "surf"))), "unknown type is an error");

S = load("gradient_features.mat");
chk(numel(S.f_sift) == 600 && numel(S.f_dsift) == 600 && numel(S.f_hog) == 600, ...
    "gradient_features.mat holds 600 cells of each type");
probe = [1 300 577];
ok = true;
for i = probe
    I = imread(cfg.files(i));
    ok = ok && isequal(single(S.f_sift{i}), getImageFeatures(I, struct("type", "sift"))) ...
        && isequal(single(S.f_dsift{i}), getImageFeatures(I, struct("type", "dsft"))) ...
        && isequal(S.f_hog{i}, getImageFeatures(I, struct("type", "hogf")));
end
chk(ok, "images 1, 300 and 577 reproduce from the source images");

fprintf("\n--- Q2: getFisherVectorModel ---\n");
% A known mixture of three well separated Gaussians in 4 dimensions.
muTrue = [0 0 0 0; 8 0 0 0; 0 8 0 0]';
sdTrue = [1 1 1 1; 0.5 2 1 1; 1 1 0.5 2]';
wTrue = [0.5 0.3 0.2];
n = 30000;
comp = randsample(3, n, true, wTrue);
X = muTrue(:, comp)' + randn(n, 4) .* sdTrue(:, comp)';
[g, A, lat] = getFisherVectorModel(X, 4, 3);
% The model is fitted to f*A, so A*mean takes it back to the original space.
m = A(:, 1:4) * g.mean;
[~, ord] = min(pdist2(muTrue', m'), [], 2);
chk(max(abs(m(:, ord) - muTrue), [], "all") < 0.1, "recovers the means of a known mixture");
chk(max(abs(g.prior(ord) - wTrue)) < 0.02, "recovers its weights");
chk(abs(sum(g.prior) - 1) < 1e-12 && all(g.var(:) > 0), "priors sum to 1, variances positive");
chk(norm(A' * A - eye(4)) < 1e-10, "A is orthonormal");
C = cov(X);
chk(max(abs(sort(eig(C), "descend") - lat)) < 1e-8, "lat are the eigenvalues of cov(f)");
chk(issorted(lat, "descend"), "eigenvalues sorted largest first");
chk(threw(@() getFisherVectorModel(X, 5, 2)), "kd larger than d is an error");

M = load("gmm_models.mat");
ok = true;
for t = 1:3
    for a = 1:2
        for b = 1:3
            g = M.models{t, a, b};
            ok = ok && isequal(size(g.mean), [cfg.kdList(a) cfg.ncList(b)]) ...
                && isequal(size(g.var), size(g.mean)) ...
                && abs(sum(g.prior) - 1) < 1e-9 && all(g.var(:) > 0);
        end
    end
end
chk(ok, "all 18 saved models have the right sizes and valid parameters");

fprintf("\n--- Q3: getFisherVector ---\n");
kd = 16; nc = 32;
g = M.models{1, 1, 2};
gS = M.models{1, 1, 1};
f = double(S.f_sift{5});
fv = getFisherVector(f, M.A{1}, gS, kd, nc, Normalization="none");
% The same thing one component at a time, with mvnpdf.
x = f * M.A{1}(:, 1:kd);
lik = zeros(size(x, 1), nc);
for k = 1:nc
    lik(:, k) = gS.prior(k) * mvnpdf(x, gS.mean(:, k)', diag(gS.var(:, k)));
end
q = lik ./ sum(lik, 2);
ref = zeros(2*kd, nc);
for k = 1:nc
    z = (x - gS.mean(:, k)') ./ sqrt(gS.var(:, k)');
    ref(1:kd, k) = sum(q(:, k) .* z, 1)' / (size(x, 1) * sqrt(gS.prior(k)));
    ref(kd+1:end, k) = sum(q(:, k) .* (z.^2 - 1), 1)' / ...
        (size(x, 1) * sqrt(2 * gS.prior(k)));
end
chk(max(abs(fv(:) - ref(:))) < 1e-9 * max(abs(ref(:))), ...
    "matches a loop over components with mvnpdf");
fvI = getFisherVector(f, M.A{1}, gS, kd, nc);
chk(numel(fvI) == 2*kd*nc && abs(norm(fvI) - 1) < 1e-12, "improved FV is 2*kd*nc long, unit norm");
chk(max(abs(fvI - sign(fv) .* sqrt(abs(fv)) / norm(sqrt(abs(fv))))) < 1e-12, ...
    "improved = signed square root, then L2");
chk(max(abs(getFisherVector(f(randperm(size(f, 1)), :), M.A{1}, gS, kd, nc) - fvI)) ...
    < 1e-12, "the order of the features does not matter");
chk(all(getFisherVector(zeros(0, 128), M.A{1}, gS, kd, nc) == 0), "no features gives zeros");

% Features drawn from the GMM itself carry no gradient, so the raw Fisher
% vector must shrink toward zero as 1/sqrt(n).
n = 200000;
comp = randsample(nc, n, true, gS.prior);
xs = gS.mean(:, comp)' + randn(n, kd) .* sqrt(gS.var(:, comp))';
Ainv = M.A{1};
fs = xs * Ainv(:, 1:kd)';                    % back to 128-d, so f*A recovers xs
fv0 = getFisherVector(fs, Ainv, gS, kd, nc, Normalization="none");
chk(max(abs(fv0)) < 15 / sqrt(n), sprintf( ...
    "features sampled from the GMM give a near zero FV (max %.4f)", max(abs(fv0))));

fprintf("\n--- Q4: getVLAD ---\n");
xr = f * M.A{1}(:, 1:kd);
vl = getVLAD(xr, gS, kd, nc, Normalization="none");
refV = zeros(kd, nc);
for i = 1:size(xr, 1)
    dist = sum((xr(i, :)' - gS.mean).^2, 1);
    [~, k] = min(dist);
    refV(:, k) = refV(:, k) + xr(i, :)' - gS.mean(:, k);
end
chk(max(abs(vl(:) - refV(:))) < 1e-9 * max(abs(refV(:))), ...
    "matches a feature by feature loop");
chk(abs(norm(getVLAD(xr, gS, kd, nc)) - 1) < 1e-12, "improved VLAD has unit norm");
vi = reshape(getVLAD(xr, gS, kd, nc, Normalization="intra"), kd, nc);
blockNorm = vecnorm(vi, 2, 1);
used = any(refV ~= 0, 1);
chk(max(abs(blockNorm(used) - blockNorm(find(used, 1)))) < 1e-12, ...
    "intra: every used centroid block has the same norm");
chk(threw(@() getVLAD(f, gS, kd, nc)), "unreduced 128-d input is an error");

fprintf("\n--- hw3_loo and the saved results ---\n");
Xr = randn(90, 5) + repelem(3 * eye(3, 5), 30, 1);
yr = repelem((1:3)', 30);
[accR, cmR, predR] = hw3_loo(Xr, yr);
Dm = pdist2(Xr, Xr); Dm(1:91:end) = Inf;
[~, nn] = min(Dm, [], 2);
chk(isequal(predR, yr(nn)), "knnclassify LOO equals the argmin of the distance matrix");
chk(abs(accR - mean(predR == yr)) < 1e-12 && sum(cmR(:)) == 90, ...
    "accuracy is sum(diag(cm)) / sum(cm(:))");

Q3 = load("hw3_q3_results.mat");
t = 2; a = 1; b = 2;
X = zeros(600, 2*cfg.kdList(a)*cfg.ncList(b));
for i = 1:600
    X(i, :) = getFisherVector(S.f_dsift{i}, M.A{t}, M.models{t, a, b}, ...
        cfg.kdList(a), cfg.ncList(b));
end
[acc, ~, pred] = hw3_loo(X, cfg.labels);
chk(isequal(pred, Q3.predFV{t, a, b}) && acc == Q3.accFV(t, a, b), ...
    sprintf("dense SIFT FV (16, 64) reproduces: %.3f", acc));
Dm = pdist2(X, X); Dm(1:601:end) = Inf;
[~, nn] = min(Dm, [], 2);
chk(isequal(pred, cfg.labels(nn)), "and agrees with a plain nearest neighbor search");
chk(all(cellfun(@(c) sum(c(:)), Q3.cmFV) == 600, "all"), "every confusion matrix counts 600 images");

if isfile("hw3_q4_results.mat")
    Q4 = load("hw3_q4_results.mat");
    kd = cfg.kdList(a); nc = cfg.ncList(b);
    X = zeros(600, kd*nc);
    for i = 1:600
        X(i, :) = getVLAD(double(S.f_hog{i}) * M.A{3}(:, 1:kd), ...
            M.models{3, a, b}, kd, nc);
    end
    acc = hw3_loo(X, cfg.labels);
    chk(acc == Q4.accVLAD(3, a, b), sprintf("HoG VLAD (16, 64) reproduces: %.3f", acc));
end

function tf = threw(f)
%threw - True when calling f raises an error
tf = false;
try
    f();
catch
    tf = true;
end
end
