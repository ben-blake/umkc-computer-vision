%hw1_verify4 - Check the specific factual claims made in the report
%   Each block corresponds to a sentence in the write-up that asserts
%   something checkable about the data or the metric.
%
%   See also hw1_verify3, getHSVHist

%   Homework 1 - verification harness, part 4.

clear; clc;
[files, labels, classes] = hw1_config();
E = load("hsv_hist64.mat"); K = load("hsv_hist64_kld.mat");
ct = E.color_table;

fprintf("--- Claim: histogram spread statistics ---\n");
nbE = mean(sum(E.hsv_hist > 0, 2)); nbK = mean(sum(K.hsv_hist > 0, 2));
enE = mean(-sum(E.hsv_hist .* log2(E.hsv_hist + eps), 2));
enK = mean(-sum(K.hsv_hist .* log2(K.hsv_hist + eps), 2));
fprintf("  non-empty bins: euc %.1f, kld %.1f (report: 46.3, 54.0)\n", nbE, nbK);
fprintf("  entropy bits:   euc %.2f, kld %.2f (report: 3.67, 4.11)\n", enE, enK);

fprintf("\n--- Claim: KL discards brightness ---\n");
% Pairs identical in H and S but differing in V.
hs = [0.35 0.40];
vA = 0.25; vB = 0.85;
pxA = [hs vA]; pxB = [hs vB];
a = 1e-6;
Q = (ct + a) ./ sum(ct + a, 2);
kl = @(p) sum(((p+a)./sum(p+a)) .* log(((p+a)./sum(p+a)) ./ Q), 2);
[~, binA] = min(kl(pxA)); [~, binB] = min(kl(pxB));
fprintf("  V = %.2f -> bin %d ; V = %.2f -> bin %d ; same bin: %d\n", ...
    vA, binA, vB, binB, binA == binB);
% Now a pure rescaling of the whole triple, which KL cannot see.
pxC = pxA * 0.5;
[~, binC] = min(kl(pxC));
fprintf("  triple scaled by 0.5 -> bin %d ; same bin as original: %d\n", ...
    binC, binC == binA);
% How often does V-only variation actually collapse, across the cube?
rng(0); nTest = 4000;
base = rand(nTest, 2);
v1 = rand(nTest, 1)*0.4; v2 = v1 + 0.5;
P1 = [base v1]; P2 = [base v2];
assign = @(P) begin_assign(P, Q, a);
b1 = assign(P1); b2 = assign(P2);
fprintf("  V shifted by +0.5, same bin in %.1f%% of %d random pixels\n", ...
    100*mean(b1 == b2), nTest);
% Compare with the Euclidean metric for reference.
e1 = knnbin(P1, ct); e2 = knnbin(P2, ct);
fprintf("  same test under euclidean: %.1f%% collapse\n", 100*mean(e1 == e2));

fprintf("\n--- Claim: confusion spread of basketball_court ---\n");
n_class = 15; n_img = 40; test_set = 1:10;
test_idx = reshape(((0:n_class-1)'*n_img + test_set)', [], 1);
train_idx = setdiff(1:n_class*n_img, test_idx)';
S = load("hsv_hist128.mat");
rec = knnclassify(S.hsv_hist(test_idx,:), S.hsv_hist(train_idx,:), ...
    S.labels(train_idx), 1);
yte = S.labels(test_idx);
for k = [4 8 6]
    r = rec(yte == k);
    fprintf("  %-18s spread over %d distinct predicted classes (%d wrong)\n", ...
        classes(k), numel(unique(r)), numel(unique(r(r ~= k))));
end

fprintf("\n--- Claim: desert and chaparral concentrate their mass ---\n");
for k = [13 7 8]
    m = mean(E.hsv_hist(labels == k, :), 1);
    s = sort(m, "descend");
    fprintf("  %-18s top-2 bins hold %.0f%% of mass, top-5 %.0f%%\n", ...
        classes(k), 100*sum(s(1:2)), 100*sum(s(1:5)));
end

fprintf("\n--- Claim: all code words are used ---\n");
for f = ["hsv_hist64" "hsv_hist128"]
    T = load(f + ".mat");
    dead = sum(sum(T.hsv_hist, 1) == 0);
    fprintf("  %-16s dead bins: %d of %d\n", f, dead, size(T.hsv_hist, 2));
end

function b = begin_assign(P, Q, a)
    Pn = (P + a) ./ sum(P + a, 2);
    D = (sum(Pn .* log(Pn), 2) - Pn * log(Q)');
    [~, b] = min(D, [], 2);
end

function b = knnbin(P, ct)
    [~, b] = min(pdist2(ct, P), [], 1);
    b = b(:);
end
