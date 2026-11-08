function [A1, eigv] = getFisherface(training_data, A0, opts)
%getFisherface - LDA face model on top of the Eigenface model
%   A1 = getFisherface(training_data, A0) computes Fisher's linear
%   discriminant of the training faces in the Eigenface space of A0.
%   training_data is a structure with the n-by-400 face matrix in faces
%   and the n-by-1 subject labels in ids, and A0 is the 400-by-d0
%   Eigenface basis from getEigenface. A1 is d0-by-d1, and a face f is
%   represented by f*A0*A1.
%
%   The columns of A1 solve the generalized eigenproblem
%       Sb a = lambda Sw a
%   where Sb and Sw are the between-class and within-class scatter of
%   the faces in the Eigenface space. They are sorted by decreasing
%   lambda, and scaled so that A1'*Sw*A1 is the identity, which makes
%   Euclidean distance in the Fisherface space measure distance in units
%   of the spread within a class.
%
%   d1 is min(d0, c - 1) for c subjects, the most directions LDA can
%   give. Computing the Eigenface model first, so that d0 is less than
%   n - c, is what keeps Sw invertible.
%
%   [A1, eigv] = getFisherface(...) also returns the d1 eigenvalues
%   lambda, the ratio of between-class to within-class variance along
%   each direction.
%
%   [...] = getFisherface(..., Dim=D) also specifies d1, at most
%   min(d0, c - 1).
%
%   [...] = getFisherface(..., Regularization=R) also adds R times the
%   mean diagonal of Sw to its diagonal before solving. The default is
%   0. A small R keeps Sw invertible when many subjects have a single
%   face, as in the local models of Q4.
%
%   Example: Fisherfaces of the first 1200 faces, d0 = 32
%       load small-face-dataset/faces-ids-n6680-m417-20x20.mat
%       train = struct("faces", faces(1:1200,:), "ids", ids(1:1200));
%       A0 = getEigenface(train, 32);
%       A1 = getFisherface(train, A0);
%       fisherfaces = A0 * A1;      % 400-by-d1, one basis image per column
%
%   See also getEigenface, getLaplacianface, eig

%   Homework 4, Q1 - Fisherface.

arguments
    training_data (1,1) struct
    A0 (:,:) double
    opts.Dim double {mustBeScalarOrEmpty, mustBeInteger, mustBePositive} = []
    opts.Regularization (1,1) double {mustBeNonnegative} = 0
end

x = double(training_data.faces) * A0;          % n-by-d0
y = training_data.ids(:);
[classes, ~, g] = unique(y);
c = numel(classes);
d0 = size(A0, 2);
if c < 2
    error("getFisherface:oneClass", "LDA needs at least two subjects.");
end

% Scatter matrices. Each face is taken relative to the mean of its own
% subject for Sw, and each subject mean relative to the overall mean,
% weighted by the subject's number of faces, for Sb.
nk = accumarray(g, 1);
member = sparse(g, 1:numel(y), 1, c, numel(y));  % c-by-n, 1 where face i is subject k
classMean = (member * x) ./ nk;                % c-by-d0
within = x - classMean(g, :);
Sw = within' * within;
between = (classMean - mean(x, 1)) .* sqrt(nk);
Sb = between' * between;
Sw = (Sw + Sw') / 2;
Sb = (Sb + Sb') / 2;
Sw = Sw + opts.Regularization * mean(diag(Sw)) * eye(d0);

d1 = min(d0, c - 1);
if ~isempty(opts.Dim)
    if opts.Dim > d1
        error("getFisherface:dimTooLarge", ...
            "Dim = %d, but at most %d directions exist.", opts.Dim, d1);
    end
    d1 = opts.Dim;
end

% The "chol" algorithm needs Sw positive definite, and it returns
% eigenvectors normalized so that A1'*Sw*A1 = I.
[V, D] = eig(Sb, Sw, "chol");
[eigv, order] = sort(real(diag(D)), "descend");
A1 = V(:, order(1:d1));
eigv = eigv(1:d1);
end
