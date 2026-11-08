function [A2, W, eigv, h] = getLaplacianface(training_data, A0, opts)
%getLaplacianface - Laplacian face model (LPP) on top of the Eigenface model
%   A2 = getLaplacianface(training_data, A0) computes the locality
%   preserving projection of the training faces in the Eigenface space
%   of A0. training_data is a structure with the n-by-400 face matrix in
%   faces and the n-by-1 subject labels in ids, and A0 is the 400-by-d0
%   Eigenface basis from getEigenface. A2 is d0-by-d2, and a face f is
%   represented by f*A0*A2.
%
%   The affinity W is the supervised heat kernel of the course code:
%       W(i,j) = exp(-h * ||x_i - x_j||)   if faces i and j are the same subject
%       W(i,j) = 0                         otherwise
%   with h = -log(0.15) / mean distance, so that a pair at the mean
%   distance has an affinity of 0.15. With D the diagonal matrix of the
%   row sums of W and L = D - W the graph Laplacian, the columns of A2
%   solve the generalized eigenproblem
%       X' L X a = lambda X' D X a
%   for the d2 smallest lambda, which keeps faces with a high affinity
%   close together. X holds the training faces in the Eigenface space
%   with their mean subtracted, as in the provided LPP.m. The columns
%   are scaled so that A2'*X'*D*X*A2 is the identity. Centering only
%   shifts every face by the same amount, so f*A0*A2 can still be used
%   to represent a face.
%
%   [A2, W] = getLaplacianface(...) also returns the n-by-n affinity W as
%   a sparse matrix. It only has entries within a subject.
%
%   [A2, W, eigv] = getLaplacianface(...) also returns the d2 eigenvalues
%   lambda, smallest first.
%
%   [A2, W, eigv, h] = getLaplacianface(...) also returns the kernel
%   width h.
%
%   [...] = getLaplacianface(..., Dim=D) also specifies d2. The default
%   is min(d0, c - 1) for c subjects, the same as getFisherface.
%
%   [...] = getLaplacianface(..., Affinity=0.15) also specifies the
%   affinity of a pair of faces at the mean distance, which sets h.
%
%   Example: Laplacianfaces of the first 1200 faces, d0 = 32
%       load small-face-dataset/faces-ids-n6680-m417-20x20.mat
%       train = struct("faces", faces(1:1200,:), "ids", ids(1:1200));
%       A0 = getEigenface(train, 32);
%       [A2, W] = getLaplacianface(train, A0);
%       spy(W)                      % block diagonal, one block per subject
%
%   See also getEigenface, getFisherface, eig

%   Homework 4, Q1 - Laplacianface.

arguments
    training_data (1,1) struct
    A0 (:,:) double
    opts.Dim double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
    opts.Affinity (1,1) double {mustBeInRange(opts.Affinity, 0, 1, "exclusive")} = 0.15
end

x = double(training_data.faces) * A0;          % n-by-d0
y = training_data.ids(:);
n = numel(y);
c = numel(unique(y));
d0 = size(A0, 2);

% The kernel width comes from the mean distance over all pairs, as in the
% course code. The full distance matrix is n-by-n, so it is built a block
% of rows at a time and only its sum is kept.
total = 0;
block = 2000;
for i = 1:block:n
    rows = i:min(i + block - 1, n);
    total = total + sum(pdist2(x(rows, :), x), "all");
end
h = -log(opts.Affinity) / (total / n^2);

% Supervised affinity: only pairs of the same subject are connected, so
% W is block diagonal and is built one subject at a time.
[I, J, V] = deal(cell(c, 1));
[~, ~, g] = unique(y);
for k = 1:c
    idx = find(g == k);
    [jj, ii] = meshgrid(idx, idx);
    I{k} = ii(:);
    J{k} = jj(:);
    V{k} = reshape(exp(-h * pdist2(x(idx, :), x(idx, :))), [], 1);
end
W = sparse(vertcat(I{:}), vertcat(J{:}), vertcat(V{:}), n, n);

% The faces are centered first, as the provided LPP.m does. Without it
% the problem has an almost trivial solution, a direction along which
% every face projects to nearly the same value, since L times a constant
% is zero, and that direction would come first while carrying no
% information about who is who.
xc = x - mean(x, 1);
deg = full(sum(W, 2));
XDX = xc' * (deg .* xc);
XLX = XDX - xc' * (W * xc);
XDX = (XDX + XDX') / 2;
XLX = (XLX + XLX') / 2;

d2 = min(d0, c - 1);
if ~isempty(opts.Dim)
    if opts.Dim > d0
        error("getLaplacianface:dimTooLarge", ...
            "Dim = %d, but the Eigenface space has only %d.", opts.Dim, d0);
    end
    d2 = opts.Dim;
end

[V, D] = eig(XLX, XDX, "chol");
[eigv, order] = sort(real(diag(D)), "ascend");
A2 = V(:, order(1:d2));
eigv = eigv(1:d2);
end
