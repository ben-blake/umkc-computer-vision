function ap = getAveragePrecision(hit, opts)
%getAveragePrecision - Average precision of a ranked list
%   ap = getAveragePrecision(hit) returns the mean of the precision at
%   each rank of the logical vector hit that holds a relevant item, where
%   hit(k) is true when the item retrieved at rank k is relevant. A list
%   with no relevant item scores 0. Passing the top K of a ranking gives
%   AP at K as the example in the assignment computes it: a single
%   relevant item at rank 2 of 5 gives 1/2.
%
%   ap = getAveragePrecision(hit, NumRelevant=Q) divides the sum of
%   those precisions by Q instead of by the number of hits, which is the
%   definition on the course slide: every relevant item that was not
%   retrieved counts as a precision of 0. With the whole ranking and Q
%   the number of relevant items, this is the usual AP.
%
%   Example: One hit at rank 2, under both definitions
%       getAveragePrecision([0 1 0 0 0])                   % 0.5
%       getAveragePrecision([0 1 0 0 0], NumRelevant=5)    % 0.1
%
%   See also cumsum, hw4_q3_retrieval

%   Homework 4, Q3 - average precision.

arguments
    hit {mustBeNumericOrLogical, mustBeVector}
    opts.NumRelevant double {mustBeScalarOrEmpty, mustBePositive} = []
end

hit = logical(hit(:)');
if ~any(hit)
    ap = 0;
    return
end
precision = cumsum(hit) ./ (1:numel(hit));
if isempty(opts.NumRelevant)
    ap = mean(precision(hit));
else
    ap = sum(precision(hit)) / opts.NumRelevant;
end
end
