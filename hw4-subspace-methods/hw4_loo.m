function [acc, cm, pred, labels] = hw4_loo(X, y)
%hw4_loo - Leave-one-out top-1 recognition by the nearest neighbor
%   [acc, cm] = hw4_loo(X, y) recognizes every row of the N-by-d feature
%   matrix X as the label of its nearest other row, by Euclidean
%   distance, and returns the accuracy and the confusion matrix. cm is
%   K-by-K for the K labels in y, true label down the rows and predicted
%   label across, and acc = sum(diag(cm)) / sum(cm(:)).
%
%   [acc, cm, pred] = hw4_loo(...) also returns the N-by-1 predictions.
%
%   [acc, cm, pred, labels] = hw4_loo(...) also returns the labels in the
%   order of the rows and columns of cm.
%
%   Example: Accuracy of random features, which should be near chance
%       [acc, cm] = hw4_loo(randn(60, 8), repelem((1:3)', 20))
%
%   See also pdist2, confusionmat, hw4_q2_recognition

%   Homework 4 - shared recognition protocol.

arguments
    X (:,:) double
    y (:,1) double
end

D = pdist2(X, X);
D(1:size(D, 1) + 1:end) = Inf;          % never match a face with itself
[~, nearest] = min(D, [], 2);
pred = y(nearest);
labels = unique(y);
cm = confusionmat(y, pred, Order=labels);
acc = sum(diag(cm)) / sum(cm(:));
end
