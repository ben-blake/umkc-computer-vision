function [gmm, A, lat] = getFisherVectorModel(f, kd, nc, opts)
%getFisherVectorModel - PCA and diagonal GMM for Fisher vector encoding
%   [gmm, A] = getFisherVectorModel(f, kd, nc) learns the two models that
%   getFisherVector needs from the n-by-d matrix f of local features
%   pooled over many images. A is the d-by-d matrix of principal
%   directions from pca, columns ordered by decreasing variance, so
%   f*A(:,1:kd) is the feature reduced to kd dimensions. gmm is a
%   Gaussian mixture of nc components with diagonal covariance fitted to
%   that reduced feature, in the layout vl_gmm uses:
%       gmm.mean  - kd-by-nc, one column per component
%       gmm.var   - kd-by-nc, the diagonal of each covariance
%       gmm.prior - 1-by-nc, the mixing weights, which sum to one
%
%   As in the assignment template the projection is f*A(:,1:kd) without
%   subtracting the mean. getFisherVector projects the same way, so the
%   offset is absorbed into gmm.mean and the encoding does not change.
%
%   [gmm, A, lat] = getFisherVectorModel(...) also returns the d-by-1
%   eigenvalues of the feature covariance, largest first.
%
%   [...] = getFisherVectorModel(..., Seed=S) also specifies the seed for
%   the k-means++ initialization. The default is 0.
%
%   [...] = getFisherVectorModel(..., MaxIter=M) also specifies the
%   maximum number of EM iterations. The default is 500.
%
%   [...] = getFisherVectorModel(..., Regularization=R) also specifies
%   the variance floor added to every component, as a fraction of the
%   mean variance of the reduced feature. The default is 1e-3, which
%   keeps a component from collapsing onto a few identical points.
%
%   Example: 16-dimensional, 32-component model of SIFT features
%       f = getImageFeatures(imread("NWPU-RESISC45/beach/beach_001.jpg"));
%       [gmm, A] = getFisherVectorModel(double(f), 16, 32);
%
%   See also pca, fitgmdist, getFisherVector, getVLAD

%   Homework 3, Q2 - PCA and GMM model.

arguments
    f (:,:) {mustBeNumeric}
    kd (1,1) double {mustBeInteger, mustBePositive}
    nc (1,1) double {mustBeInteger, mustBePositive}
    opts.Seed (1,1) double = 0
    opts.MaxIter (1,1) double {mustBeInteger, mustBePositive} = 500
    opts.Regularization (1,1) double {mustBeNonnegative} = 1e-3
end

f = double(f);
if kd > size(f, 2)
    error("getFisherVectorModel:kdTooLarge", ...
        "kd = %d is larger than the feature dimension %d.", kd, size(f, 2));
end

% princomp was removed from MATLAB. pca returns the same directions.
[A, ~, lat] = pca(f);
f0 = f * A(:, 1:kd);

rng(opts.Seed);
reg = opts.Regularization * mean(var(f0));
model = fitgmdist(f0, nc, CovarianceType="diagonal", ...
    RegularizationValue=reg, Start="plus", Replicates=1, ...
    Options=statset(MaxIter=opts.MaxIter));

gmm = struct();
gmm.mean = model.mu';
gmm.var = reshape(model.Sigma, kd, nc);
gmm.prior = model.ComponentProportion;
gmm.converged = model.Converged;
gmm.iterations = model.NumIterations;
gmm.negLogLikelihood = model.NegativeLogLikelihood;
end
