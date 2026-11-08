%hw4_verify - Independent checks on the Homework 4 functions and results
%   Tests getEigenface, getFisherface, getLaplacianface, getFaceCrop,
%   getAveragePrecision and getGrassmannDistance against cases whose
%   answer is known in advance and against the code provided with the
%   assignment: getLDA.m, and LPP.m from LPP.zip, which is unzipped to a
%   temporary folder. Then confirms that the saved results reproduce.
%   Prints PASS or FAIL for each check.
%
%   See also getEigenface, getFisherface, getLaplacianface,
%   getGrassmannDistance

%   Homework 4 - verification harness.

clear; clc; rng(0);
chk = @(tf, name) fprintf("%-72s %s\n", name, ...
    string(tf).replace("true","PASS").replace("false","FAIL"));
cfg = hw4_config();
here = fileparts(mfilename("fullpath"));
P = load(cfg.dataFile, "faces", "ids");
F = load(cfg.facesFile);

fprintf("--- data set and the added faces ---\n");
chk(isequal(size(F.faces), [6690 400]) && numel(unique(F.ids)) == 418, ...
    "augmented data set: 6690 faces of 418 subjects");
chk(isequal(F.faces(1:6680, :), P.faces) && isequal(F.ids(1:6680), P.ids), ...
    "the first 6680 are the provided faces, unchanged and in order");
chk(isequal(F.ids(6681:end), repmat(cfg.celebId, 10, 1)), "the last 10 are the celebrity, ID 418");
chk(isequal(size(F.selfFaces), [15 400]) && all(F.selfIds == 14387365), ...
    "15 faces of me with my student ID");
rowsMin = min([F.selfFaces; F.celebFaces], [], 2);
rowsMax = max([F.selfFaces; F.celebFaces], [], 2);
chk(all(abs(rowsMin) < 1e-12) && all(abs(rowsMax - 1) < 1e-12), ...
    "every added face is stretched to exactly [0, 1]");
k = 1;
again = getFaceCrop(imread(fullfile(cfg.selfDir, F.selfNames(k) + iExt(cfg.selfDir, F.selfNames(k)))));
chk(max(abs(again - F.selfFaces(k, :))) < 1e-12, "my first face reproduces from its photo");
chk(threw(@() getFaceCrop(zeros(300, 300))), "a blank image is an error, not a crop");

fprintf("\n--- Q1: getEigenface ---\n");
tr = struct("faces", P.faces(1:1200, :), "ids", P.ids(1:1200));
[A0, eigv] = getEigenface(tr, 32);
[V, D] = eig(cov(tr.faces));
[ev, order] = sort(diag(D), "descend");
chk(max(abs(ev - eigv)) < 1e-10 * ev(1), "eigenvalues equal eig(cov(faces))");
chk(getGrassmannDistance(A0, V(:, order(1:32))) < 1e-6, ...
    "the 32 directions span the top 32 eigenvectors of the covariance");
chk(norm(A0' * A0 - eye(32)) < 1e-10, "A0 is orthonormal");
chk(isequal(getEigenface(tr, 8), getEigenface(tr.faces, 8)), "a matrix or a structure gives the same");

fprintf("\n--- Q1: getFisherface ---\n");
x = tr.faces * A0;
[~, ~, g] = unique(tr.ids);
Sb = zeros(32); Sw = zeros(32); mx = mean(x);
for c = 1:max(g)
    xc = x(g == c, :); mc = mean(xc, 1);
    Sb = Sb + size(xc, 1) * (mc - mx)' * (mc - mx);
    Sw = Sw + (xc - mc)' * (xc - mc);
end
[A1, e1] = getFisherface(tr, A0);
resid = arrayfun(@(i) norm(Sb * A1(:, i) - e1(i) * Sw * A1(:, i)) / norm(Sb * A1(:, i)), 1:32);
chk(max(resid) < 1e-10, "every column solves Sb a = lambda Sw a, scatter from a plain loop");
chk(norm(A1' * Sw * A1 - eye(32)) < 1e-8, "A1' Sw A1 = I");
chk(issorted(e1, "descend") && all(e1 > 0), "eigenvalues positive and largest first");
addpath(fullfile(here, "small-face-dataset"));
[Aref, vref] = getLDA(x, tr.ids);
Aref = real(Aref);
residRef = arrayfun(@(i) norm(Sb * Aref(:, i) - real(vref(i, i)) * Sw * Aref(:, i)) / ...
    norm(Sb * Aref(:, i)), 1:size(Aref, 2));
fprintf("    provided getLDA.m: residuals %s, ours %s\n", mat2str(residRef, 2), mat2str(resid(1:6), 2));
rayleigh = @(A) diag(A' * Sb * A) ./ diag(A' * Sw * A);
lamOurs = rayleigh(A1(:, 1));
lamRef = rayleigh(Aref(:, 1));
fprintf("    first direction, Sb/Sw: ours %.3f, getLDA.m %.3f\n", lamOurs, lamRef);
chk(lamOurs >= lamRef && lamRef > 0.95 * lamOurs && max(residRef) > 100 * max(resid), ...
    "beats getLDA.m's eigs on the first direction, within 5%, and is exact");
chk(size(getFisherface(tr, A0, Dim=5), 2) == 5, "Dim picks the number of directions");
two = struct("faces", tr.faces(tr.ids == 1, :), "ids", tr.ids(tr.ids == 1));
chk(threw(@() getFisherface(two, A0)), "one subject is an error");

fprintf("\n--- Q1: getLaplacianface ---\n");
[A2, W, e2, h] = getLaplacianface(tr, A0);
chk(issparse(W) && nnz(W - W') == 0, "W is sparse and symmetric");
chk(full(all(diag(W) == 1)), "every face has affinity 1 with itself");
[I, J] = find(W);
chk(all(tr.ids(I) == tr.ids(J)), "W only connects faces of the same subject");
Dall = pdist2(x, x);
chk(abs(exp(-h * mean(Dall(:))) - 0.15) < 1e-12, "h puts the mean distance at an affinity of 0.15");
Wfull = exp(-h * Dall) .* (tr.ids == tr.ids');
chk(max(abs(W - Wfull), [], "all") < 1e-12, "W equals the heat kernel within a subject, built in one go");
unz = tempname;
unzip(fullfile(here, "small-face-dataset", "LPP.zip"), unz);
addpath(fullfile(unz, "LPP"));
[Alpp, elpp] = LPP(W, struct("PCARatio", 1, "ReducedDim", 31), x);
rmpath(fullfile(unz, "LPP"));
chk(max(abs((1 - e2(1:10)) - elpp(1:10))) < 1e-8, ...
    "eigenvalues equal 1 - those of the provided LPP.m");
chk(getGrassmannDistance(A2(:, 1:16), Alpp(:, 1:16)) < 1e-6, ...
    "the first 16 Laplacianfaces span the same space as LPP.m's");
xc = x - mean(x);
dg = full(sum(W, 2));
chk(norm(A2' * (xc' * (dg .* xc)) * A2 - eye(size(A2, 2))) < 1e-8, "A2' X'DX A2 = I");

fprintf("\n--- Q3: getAveragePrecision ---\n");
chk(getAveragePrecision([0 1 0 0 0]) == 0.5, "the assignment's example: one hit at rank 2 is 50%");
chk(getAveragePrecision([1 1 0 0 0]) == 1, "hits at ranks 1 and 2 are 100%");
chk(abs(getAveragePrecision([1 0 1 0 0]) - (1 + 2/3) / 2) < 1e-12, "hits at 1 and 3: (1 + 2/3) / 2");
chk(getAveragePrecision([0 0 0 0 0]) == 0, "no hit is 0");
chk(abs(getAveragePrecision([0 1 0 0 0], NumRelevant=5) - 0.1) < 1e-12, ...
    "slide definition: the same list with 5 relevant is 0.5 / 5 = 0.1");
chk(abs(getAveragePrecision([1 1 0 1], NumRelevant=4) - (1 + 1 + 3/4) / 4) < 1e-12, ...
    "slide definition: an unretrieved relevant item counts as 0");

fprintf("\n--- Q4: getGrassmannDistance ---\n");
Q = orth(randn(50, 4));
R = orth(randn(4));
chk(getGrassmannDistance(Q, Q * R) < 1e-7, "the same subspace in another basis: 0");
chk(getGrassmannDistance(Q, [Q, randn(50, 2)]) < 1e-7, "a subspace inside a larger one: 0");
chk(abs(getGrassmannDistance([1 0 0]', [0 1 0]') - 1) < 1e-12, "orthogonal lines: 1");
t = 0.3;
chk(abs(getGrassmannDistance([1 0]', [cos(t) sin(t)]') - sin(t)) < 1e-12, "two lines at angle t: sin t");
[dd, th] = getGrassmannDistance([1 0; 0 1; 0 0], [1 0; 0 0; 0 1]);
chk(abs(dd - 1) < 1e-12 && max(abs(th - [0; pi/2])) < 1e-7, "planes sharing one direction: angles 0 and 90");

fprintf("\n--- the saved results reproduce ---\n");
M = load("hw4_models.mat");
Q2 = load("hw4_q2_results.mat");
train = struct("faces", F.faces(M.trainIdx, :), "ids", F.ids(M.trainIdx));
chk(~any(ismember(F.ids(M.trainIdx), F.ids(Q2.testIdx))), "no training subject is a test subject");
B0 = getEigenface(train, cfg.d0);
chk(max(abs(abs(B0) - abs(M.A0)), [], "all") < 1e-10, "A0 reproduces from the training faces");
X = F.faces(Q2.testIdx, :); y = F.ids(Q2.testIdx);
[accF, ~, predF] = hw4_loo(X * M.A0 * M.A1, y);
Dm = pdist2(X * M.A0 * M.A1, X * M.A0 * M.A1);
Dm(1:size(Dm, 1) + 1:end) = Inf;
[~, nn] = min(Dm, [], 2);
chk(isequal(predF, y(nn)) && accF == Q2.acc(2), ...
    sprintf("Fisherface accuracy %.3f reproduces, and is a plain nearest neighbor", accF));
chk(all(cellfun(@(c) sum(c(:)), Q2.cm) == numel(y)), "every confusion matrix counts every test face");
Q3 = load("hw4_q3_results.mat");
chk(abs(mean(Q3.apK(:, 1)) - Q3.mapK(1)) < 1e-12 && all(Q3.apK(:) >= 0 & Q3.apK(:) <= 1), ...
    "mAP is the mean of the per-query AP, every AP in [0, 1]");
Q4 = load("hw4_q4_results.mat");
rng(cfg.seed);
q_offs = randperm(cfg.nLocal);
chk(isequal(q_offs(1:cfg.nQuery), Q4.q_offs), "the 20 queries are the assignment's randperm draw, seed 0");

rmpath(fullfile(here, "small-face-dataset"));

function tf = threw(f)
%threw - True when calling f raises an error
tf = false;
try
    f();
catch
    tf = true;
end
end

function ext = iExt(folder, name)
%iExt - Extension of the file called name in folder
d = dir(fullfile(folder, name + ".*"));
[~, ~, ext] = fileparts(d(1).name);
ext = string(ext);
end
