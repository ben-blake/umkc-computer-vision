%hw1_verify2 - Verify the classifier, the split, and the significance
%   Cross-checks knnclassify against fitcknn and against a raw pdist2
%   argmin, confirms the train and test sets are disjoint, and attaches
%   confidence intervals and McNemar tests to the accuracy differences
%   the report draws conclusions from.
%
%   See also hw1_verify, knnclassify, hw1_part3_recog

%   Homework 1 - verification harness, part 2.

clear; clc;
chk = @(tf, name) fprintf("%-58s %s\n", name, ...
    string(tf).replace("true", "PASS").replace("false", "FAIL"));

featSets = ["hsv_hist64" "hsv_hist64_kld" "hsv_hist128" "hsv_hist128_kld"];
n_class = 15; n_img = 40; test_set = 1:10;
test_idx = reshape(((0:n_class-1)'*n_img + test_set)', [], 1);
train_idx = setdiff(1:n_class*n_img, test_idx)';

fprintf("--- Split integrity ---\n");
chk(numel(test_idx) == 150 && numel(train_idx) == 450, "150 test, 450 train");
chk(isempty(intersect(test_idx, train_idx)), "train and test are disjoint");
chk(isequal(sort([test_idx; train_idx]), (1:600)'), "together they cover all 600");
load hsv_hist64.mat labels
chk(all(accumarray(labels(test_idx), 1) == 10), "10 test images per class");
chk(all(accumarray(labels(train_idx), 1) == 30), "30 train images per class");
chk(isequal(labels(test_idx), repelem((1:15)', 10)), ...
    "test labels run class by class, matching rec order");

fprintf("\n--- knnclassify against independent implementations ---\n");
for f = 1:numel(featSets)
    S = load(featSets(f) + ".mat");
    Xtr = S.hsv_hist(train_idx, :); ytr = S.labels(train_idx);
    Xte = S.hsv_hist(test_idx, :);  yte = S.labels(test_idx);

    ours = knnclassify(Xte, Xtr, ytr, 1);

    % fitcknn, the supported modern equivalent.
    mdl = fitcknn(Xtr, ytr, NumNeighbors=1, Distance="euclidean");
    theirs = predict(mdl, Xte);

    % Raw argmin over a pdist2 matrix, no classifier object at all.
    [~, nn] = min(pdist2(Xtr, Xte), [], 1);
    raw = ytr(nn);

    agree1 = mean(ours == theirs);
    agree2 = mean(ours == raw);
    fprintf("  %-16s acc %.4f | agrees fitcknn %.3f | agrees argmin %.3f\n", ...
        featSets(f), mean(ours == yte), agree1, agree2);
    chk(agree1 == 1 && agree2 == 1, "  identical predictions, " + featSets(f));
end

fprintf("\n--- k = 3 and k = 5 cross-check ---\n");
S = load("hsv_hist128.mat");
Xtr = S.hsv_hist(train_idx, :); ytr = S.labels(train_idx);
Xte = S.hsv_hist(test_idx, :);  yte = S.labels(test_idx);
for kk = [3 5]
    ours = knnclassify(Xte, Xtr, ytr, kk);
    mdl = fitcknn(Xtr, ytr, NumNeighbors=kk, Distance="euclidean", ...
        BreakTies="nearest");
    theirs = predict(mdl, Xte);
    fprintf("  k = %d: ours %.4f, fitcknn %.4f, agreement %.3f\n", ...
        kk, mean(ours == yte), mean(theirs == yte), mean(ours == theirs));
end

fprintf("\n--- Leave-one-out cross-check, hsv_hist64 ---\n");
S = load("hsv_hist64.mat");
H = S.hsv_hist; y = S.labels;
D = pdist2(H, H);
D(1:size(D,1)+1:end) = Inf;          % exclude self
[~, nn] = min(D, [], 1);
looRaw = mean(y(nn) == y);
fprintf("  matrix-based LOO accuracy: %.4f (script reported 0.5250)\n", looRaw);
chk(abs(looRaw - 0.525) < 1e-12, "  LOO reproduces exactly");

fprintf("\n--- Statistical significance, n = 150 ---\n");
[~, ci] = binofit(round(0.5133*150), 150);
fprintf("  51.3%% has 95%% CI [%.1f%%, %.1f%%] -- width %.1f points\n", ...
    100*ci(1), 100*ci(2), 100*(ci(2)-ci(1)));

% Paired McNemar tests on the same 150 test images.
function p = mcnemar(a, b, y)
    b1 = sum((a == y) & (b ~= y));
    c1 = sum((a ~= y) & (b == y));
    p = 2 * binocdf(min(b1, c1), b1 + c1, 0.5);
    p = min(p, 1);
    fprintf("      discordant: %d vs %d, p = %.3f\n", b1, c1, p);
end

S64 = load("hsv_hist64.mat"); S128 = load("hsv_hist128.mat");
y = S64.labels(test_idx);
p64  = knnclassify(S64.hsv_hist(test_idx,:),  S64.hsv_hist(train_idx,:),  S64.labels(train_idx), 1);
p128 = knnclassify(S128.hsv_hist(test_idx,:), S128.hsv_hist(train_idx,:), S128.labels(train_idx), 1);
fprintf("  K = 64 vs K = 128 (51.3%% vs 53.3%%):\n");
mcnemar(p128, p64, y);

pChi = knnclassify(S128.hsv_hist(test_idx,:), S128.hsv_hist(train_idx,:), ...
    S128.labels(train_idx), 1, "chisq");
fprintf("  euclidean vs chi-square on hsv_hist128 (53.3%% vs 58.0%%):\n");
mcnemar(pChi, p128, y);

Sk = load("hsv_hist64_kld.mat");
pKld = knnclassify(Sk.hsv_hist(test_idx,:), Sk.hsv_hist(train_idx,:), ...
    Sk.labels(train_idx), 1);
fprintf("  euc vs kld assignment at K = 64 (51.3%% vs 48.7%%):\n");
mcnemar(p64, pKld, y);
