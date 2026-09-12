function [class] = knnclassify(sample, training, group, k, distance, rule)
%knnclassify - Classify data using the nearest neighbor rule
%   CLASS = knnclassify(SAMPLE, TRAINING, GROUP) classifies each row of
%   SAMPLE into one of the groups in GROUP using the nearest neighbor
%   found among the rows of TRAINING. SAMPLE and TRAINING must have the
%   same number of columns, and GROUP must have one label per row of
%   TRAINING. CLASS has one label per row of SAMPLE, in the same form as
%   GROUP.
%
%   CLASS = knnclassify(SAMPLE, TRAINING, GROUP, K) also specifies the
%   number of nearest neighbors used in the majority vote. The default
%   is 1.
%
%   CLASS = knnclassify(SAMPLE, TRAINING, GROUP, K, DISTANCE) also
%   specifies the distance metric, which is passed to pdist2. DISTANCE
%   must be:
%       "euclidean"   - (default) Euclidean distance.
%       "cityblock"   - Sum of absolute differences.
%       "cosine"      - One minus the cosine of the included angle.
%       "correlation" - One minus the sample correlation.
%       "hamming"     - Fraction of differing coordinates.
%       "chisq"       - Chi-square distance between histograms.
%
%   CLASS = knnclassify(SAMPLE, TRAINING, GROUP, K, DISTANCE, RULE) also
%   specifies how ties in the vote are broken. RULE must be:
%       "nearest"   - (default) Use the class of the nearest tied
%                     neighbor.
%       "random"    - Pick at random among the tied classes.
%       "consensus" - Leave the sample unclassified unless all K
%                     neighbors agree. Unclassified rows are NaN for
%                     numeric groups and empty otherwise.
%
%   Example: Classify three rows against a two-class training set
%       tr = [0 0; 0.1 0; 1 1; 1.1 1];
%       cl = knnclassify([0.2 0; 0.9 1; 0 0.1], tr, [1 1 2 2], 3);
%
%   Note: knnclassify was removed from Bioinformatics Toolbox after
%   R2013b. This file reproduces its interface on top of pdist2 so that
%   the assignment code runs unchanged. New code should use fitcknn
%
%   See also getHSVHist, pdist2, fitcknn, ClassificationKNN

%   Homework 1, Part 3 - drop-in replacement for the removed function.

arguments
    sample double
    training double
    group
    k (1,1) double {mustBeInteger, mustBePositive} = 1
    distance (1,1) string = "euclidean"
    rule (1,1) string {mustBeMember(rule, ...
        ["nearest", "random", "consensus"])} = "nearest"
end

if size(sample, 2) ~= size(training, 2)
    error("knnclassify:sizeMismatch", ...
        "SAMPLE and TRAINING must have the same number of columns.");
end
if numel(group) ~= size(training, 1)
    error("knnclassify:groupMismatch", ...
        "GROUP must have one label per row of TRAINING.");
end

[uGroup, ~, gIdx] = unique(group(:));
nClass = numel(uGroup);
nSample = size(sample, 1);
k = min(k, size(training, 1));

if distance == "chisq"
    D = iChiSquare(training, sample);
else
    D = pdist2(training, sample, distance);
end

% Rank the training rows for every sample and keep the K closest.
[~, order] = sort(D, 1);
nnIdx = order(1:k, :);                  % K-by-nSample
nnLabel = gIdx(nnIdx);
if k == 1
    nnLabel = reshape(nnLabel, [1 nSample]);
end

% Tally the votes of the K neighbors, one row per class.
votes = zeros(nClass, nSample);
for c = 1:nClass
    votes(c, :) = sum(nnLabel == c, 1);
end
[maxVote, winner] = max(votes, [], 1);

% Resolve the samples whose vote is tied between two or more classes.
tied = sum(votes == maxVote, 1) > 1;
unresolved = false(1, nSample);
for j = find(tied)
    tiedClass = find(votes(:, j) == maxVote(j));
    switch rule
        case "nearest"
            % nnLabel is ordered by distance, so the first neighbor
            % belonging to a tied class settles the vote.
            hit = find(ismember(nnLabel(:, j), tiedClass), 1);
            winner(j) = nnLabel(hit, j);
        case "random"
            winner(j) = tiedClass(randi(numel(tiedClass)));
        case "consensus"
            unresolved(j) = true;
    end
end
if rule == "consensus"
    unresolved = unresolved | maxVote < k;
end

class = uGroup(winner);
class = reshape(class, [nSample 1]);
if any(unresolved)
    if isnumeric(class)
        class(unresolved) = NaN;
    elseif iscell(class)
        class(unresolved) = {''};
    else
        class(unresolved) = missing;
    end
end
end

function D = iChiSquare(A, B)
%iChiSquare - Chi-square distance between rows of A and rows of B
D = zeros(size(A, 1), size(B, 1));
for j = 1:size(B, 1)
    num = (A - B(j, :)).^2;
    den = A + B(j, :) + eps;
    D(:, j) = 0.5 * sum(num ./ den, 2);
end
end
