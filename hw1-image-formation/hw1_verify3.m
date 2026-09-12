%hw1_verify3 - Leave-one-out analysis with adequate statistical power
%   The 10/30 split has only 150 test images, so its 95 percent
%   confidence interval is about 16 points wide and no difference in the
%   report reaches significance on it. Leave one out uses all 600 images
%   as test cases, which is the only protocol here with the power to
%   separate the conditions. This script recomputes every comparison
%   under LOO and attaches McNemar tests at n = 600.
%
%   See also hw1_verify2, hw1_part3_recog

%   Homework 1 - verification harness, part 3.

clear; clc;
featSets = ["hsv_hist64" "hsv_hist64_kld" "hsv_hist128" "hsv_hist128_kld"];
metrics = ["euclidean" "cityblock" "chisq"];

function D = chisqMat(H)
    n = size(H, 1);
    D = zeros(n);
    for j = 1:n
        D(:, j) = 0.5 * sum((H - H(j, :)).^2 ./ (H + H(j, :) + eps), 2);
    end
end

acc = zeros(numel(featSets), numel(metrics));
pred = cell(numel(featSets), numel(metrics));
for f = 1:numel(featSets)
    S = load(featSets(f) + ".mat");
    H = S.hsv_hist; y = S.labels;
    for m = 1:numel(metrics)
        if metrics(m) == "chisq"
            D = chisqMat(H);
        else
            D = pdist2(H, H, metrics(m));
        end
        D(1:size(D,1)+1:end) = Inf;          % never match against self
        [~, nn] = min(D, [], 1);
        pred{f, m} = y(nn);
        acc(f, m) = mean(y(nn) == y);
    end
end
load hsv_hist64.mat labels classes
y = labels;

fprintf("=== LOO accuracy (%%), 1-NN, n = 600 ===\n");
disp(array2table(100*acc, RowNames=featSets, VariableNames=metrics));

% 95 percent Clopper-Pearson interval at n = 600 for scale.
[~, ci] = binofit(round(acc(1,1)*600), 600);
fprintf("At n = 600, %.1f%% has 95%% CI [%.1f%%, %.1f%%] -- width %.1f points\n\n", ...
    100*acc(1,1), 100*ci(1), 100*ci(2), 100*(ci(2)-ci(1)));

function mcnemar(a, b, y, label)
    b1 = sum((a == y) & (b ~= y));
    c1 = sum((a ~= y) & (b == y));
    p = min(2 * binocdf(min(b1, c1), b1 + c1, 0.5), 1);
    verdict = "not significant";
    if p < 0.001
        verdict = "significant, p < 0.001";
    elseif p < 0.05
        verdict = sprintf("significant, p = %.4f", p);
    else
        verdict = sprintf("not significant, p = %.3f", p);
    end
    fprintf("  %-46s %3d vs %3d  %s\n", label, b1, c1, verdict);
end

fprintf("=== Paired McNemar tests at n = 600 ===\n");
fprintf("  %-46s %s\n", "comparison (A vs B)", "A-only  B-only  verdict");
mcnemar(pred{3,1}, pred{1,1}, y, "K=128 vs K=64, euclidean");
mcnemar(pred{1,1}, pred{2,1}, y, "euc vs kld assignment, K=64");
mcnemar(pred{3,1}, pred{4,1}, y, "euc vs kld assignment, K=128");
mcnemar(pred{1,3}, pred{1,1}, y, "chi-square vs euclidean, K=64");
mcnemar(pred{3,3}, pred{3,1}, y, "chi-square vs euclidean, K=128");
mcnemar(pred{1,2}, pred{1,1}, y, "cityblock vs euclidean, K=64");
mcnemar(pred{3,2}, pred{3,1}, y, "cityblock vs euclidean, K=128");

% Per-class accuracy under LOO: 40 cases per class instead of 10.
[bestAcc, bi] = max(acc(:));
[bf, bm] = ind2sub(size(acc), bi);
fprintf("\n=== Per-class LOO accuracy (%%), best config: %s + %s (%.1f%%) ===\n", ...
    featSets(bf), metrics(bm), 100*bestAcc);
pc = zeros(15, 2);
for k = 1:15
    pc(k,1) = 100*mean(pred{1,1}(y == k) == k);      % baseline: 64 + euc
    pc(k,2) = 100*mean(pred{bf,bm}(y == k) == k);    % best config
end
disp(array2table(pc, RowNames=classes, ...
    VariableNames=["K64_euclidean" "best_config"]));
fprintf("Chance = 6.7%%. Classes above 70%%: %d of 15. At or below 20%%: %d of 15.\n", ...
    sum(pc(:,2) >= 70), sum(pc(:,2) <= 20));
