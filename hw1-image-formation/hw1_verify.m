%hw1_verify - Independent verification of the Part 1 results
%   Re-derives every reported quantity with an implementation that does
%   not share code with the pipeline under test, and checks the data set
%   and the train/test split for the errors that would silently inflate
%   accuracy. Prints PASS or FAIL for each check.
%
%   See also getHSVHist, knnclassify, hw1_part3_recog

%   Homework 1 - verification harness.

clear; clc;
pass = 0; fail = 0;
function_handle_check = @(tf, name) fprintf("%-58s %s\n", name, ...
    string(tf).replace("true", "PASS").replace("false", "FAIL"));
chk = @(tf, name) function_handle_check(tf, name);

[files, labels, classes] = hw1_config();
load color_table64.mat color_table
ct = color_table;
M = 256*256;

fprintf("--- Data set integrity ---\n");
chk(numel(files) == 600, "600 image paths");
chk(numel(unique(files)) == 600, "no duplicate images");
chk(all(isfile(files)), "every file exists on disk");
chk(isequal(labels, repelem((1:15)', 40)), "labels are 40 of each class 1..15");
chk(all(contains(files(1:40), "airplane")), "first 40 rows are class 1");
chk(all(contains(files(561:600), "freeway")), "last 40 rows are class 15");
% The selection must be the numerically first 40 files of each class.
d = dir(fullfile(fileparts(files(1)), "*.jpg"));
firstNames = sort(string({d.name}'));
chk(all(contains(files(1:40), firstNames(1:40))), "images 001-040 selected");

fprintf("\n--- getHSVHist correctness ---\n");
im = imread(files(7));
h = getHSVHist(im, ct, "euc");
chk(abs(sum(h) - 1) < 1e-12, "euc histogram sums to 1");
chk(numel(h) == 64 && all(h >= 0), "euc histogram is a 64-bin distribution");

% Brute force nearest-code-word assignment, no pdist2 involved.
X = reshape(rgb2hsv(im), [], 3);
dsq = zeros(M, 64);
for k = 1:64
    dsq(:, k) = sum((X - ct(k, :)).^2, 2);
end
[~, bf] = min(dsq, [], 2);
hbf = accumarray(bf, 1, [64 1])' / M;
chk(max(abs(h - hbf)) < 1e-12, "euc matches brute-force assignment");

% A constant image of one code word's color must put all mass in it.
target = 20;
rgbConst = hsv2rgb(reshape(ct(target, :), [1 1 3]));
imConst = repmat(im2uint8(rgbConst), [16 16 1]);
hConst = getHSVHist(imConst, ct, "euc");
[peak, peakBin] = max(hConst);
chk(peakBin == target && peak == 1, "constant image maps to its own code word");

% KL by the direct definition sum(p.*log(p./q)), not the split form.
a = 1e-6;
P = (X + a) ./ sum(X + a, 2);
Q = (ct + a) ./ sum(ct + a, 2);
Ddirect = zeros(M, 64);
for k = 1:64
    Ddirect(:, k) = sum(P .* log(P ./ Q(k, :)), 2);
end
[~, idxKl] = min(Ddirect, [], 2);
hKlDirect = accumarray(idxKl, 1, [64 1])' / M;
hKl = getHSVHist(im, ct, "kld");
chk(max(abs(hKl - hKlDirect)) < 1e-12, "kld matches direct KL definition");
chk(all(Ddirect(:) >= -1e-9), "KL divergence is non-negative");

% The report claims KL and cross entropy pick the same bin. Test it.
[~, idxCross] = min(-P * log(Q)', [], 2);
chk(isequal(idxKl, idxCross), "KL and cross entropy give identical bins");

fprintf("\n--- Feature matrix matches the functions ---\n");
S = load("hsv_hist64.mat");
chk(isequal(size(S.hsv_hist), [600 64]), "hsv_hist64 is 600 x 64");
chk(isequal(S.labels, labels), "stored labels match hw1_config");
chk(max(abs(sum(S.hsv_hist, 2) - 1)) < 1e-12, "every row sums to 1");
chk(size(unique(S.hsv_hist, "rows"), 1) == 600, "all 600 rows are distinct");
% Recompute three scattered rows from the raw images.
probe = [7 250 588];
recomputed = zeros(3, 64);
for i = 1:3
    recomputed(i, :) = getHSVHist(imread(files(probe(i))), S.color_table, "euc");
end
chk(max(abs(recomputed - S.hsv_hist(probe, :)), [], "all") < 1e-12, ...
    "rows 7, 250, 588 reproduce from source images");
