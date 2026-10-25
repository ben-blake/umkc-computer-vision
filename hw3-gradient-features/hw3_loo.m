function [acc, cm, pred] = hw3_loo(X, labels, distance)
%hw3_loo - Leave-one-out 1-NN recognition with knnclassify
%   [acc, cm] = hw3_loo(X, labels) classifies every row of the N-by-D
%   feature matrix X with knnclassify against the other N-1 rows, and
%   returns the accuracy and the confusion matrix. cm is K-by-K for K
%   classes, with the true class down the rows and the predicted class
%   across the columns, and acc = sum(diag(cm)) / sum(cm(:)) as the
%   assignment defines it.
%
%   [acc, cm] = hw3_loo(X, labels, distance) also specifies the distance
%   knnclassify uses. The default is "euclidean", which on unit length
%   Fisher and VLAD vectors ranks neighbors the same as cosine distance.
%
%   [acc, cm, pred] = hw3_loo(...) also returns the N-by-1 predictions.
%
%   Example: Accuracy of random features, which should be near chance
%       [acc, cm] = hw3_loo(randn(60, 8), repelem((1:3)', 20))
%
%   See also knnclassify, confusionmat, hw3_q3_fisher, hw3_q4_vlad

%   Homework 3 - shared recognition protocol.

arguments
    X (:,:) double
    labels (:,1) double
    distance (1,1) string = "euclidean"
end

N = size(X, 1);
pred = zeros(N, 1);
for i = 1:N
    train = [1:i-1, i+1:N];
    pred(i) = knnclassify(X(i, :), X(train, :), labels(train), 1, distance);
end
cm = confusionmat(labels, pred, Order=unique(labels));
acc = sum(diag(cm)) / sum(cm(:));
end
