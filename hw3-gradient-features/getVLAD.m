function [fv] = getVLAD(f, gmm, kd, nc, opts)
%getVLAD - VLAD aggregation of a set of local features
%   fv = getVLAD(f, gmm, kd, nc) aggregates the n-by-kd local features f
%   of one image into a single 1-by-kd*nc vector of locally aggregated
%   descriptors. The nc columns of gmm.mean serve as the k-means
%   centroids. Every feature is assigned to its nearest centroid, and
%   the vector holds, for each centroid, the sum of the differences
%   between it and the features assigned to it.
%
%   f must already be reduced to kd dimensions with the PCA matrix of
%   getFisherVectorModel, f = double(features) * A(:,1:kd), because the
%   assignment's signature does not pass A.
%
%   fv = getVLAD(..., Normalization=NORM) also specifies the
%   normalization applied to the result. NORM must be:
%       "improved" - (default) signed square root of every entry, then
%                    unit L2 norm, the same as getFisherVector.
%       "intra"    - unit L2 norm of each centroid's block first, then
%                    of the whole vector, as in Arandjelovic and
%                    Zisserman. vl_vlad calls this NormalizeComponents.
%       "none"     - the raw residual sums.
%
%   An image with no features has a VLAD vector of zeros.
%
%   Example: VLAD of one image with the means of a 16-by-32 GMM
%       f = double(getImageFeatures(imread("NWPU-RESISC45/beach/beach_001.jpg")));
%       [gmm, A] = getFisherVectorModel(f, 16, 32);
%       fv = getVLAD(f * A(:, 1:16), gmm, 16, 32);
%       numel(fv)            % 16*32 = 512
%
%   See also getFisherVectorModel, getFisherVector, vl_vlad

%   Homework 3, Q4 - VLAD aggregation.

arguments
    f (:,:) {mustBeNumeric}
    gmm (1,1) struct
    kd (1,1) double {mustBeInteger, mustBePositive}
    nc (1,1) double {mustBeInteger, mustBePositive}
    opts.Normalization (1,1) string {mustBeMember(opts.Normalization, ...
        ["improved", "intra", "none"])} = "improved"
end

if size(f, 2) ~= kd && size(f, 1) > 0
    error("getVLAD:notReduced", ...
        "f has %d columns, not kd = %d. Reduce it first with f*A(:,1:kd).", ...
        size(f, 2), kd);
end
if ~isequal(size(gmm.mean), [kd nc])
    error("getVLAD:gmmSize", "gmm.mean must be %d-by-%d.", kd, nc);
end

if size(f, 1) == 0
    fv = zeros(1, kd*nc);
    return
end
x = double(f);
C = gmm.mean;                               % kd-by-nc

% Hard assignment to the nearest centroid.
[~, nearest] = min(pdist2(x, C'), [], 2);
assign = sparse(1:size(x, 1), nearest, 1, size(x, 1), nc);   % n-by-nc

% For centroid k: sum of x over its features minus count times c_k.
V = x' * assign - C .* full(sum(assign, 1));   % kd-by-nc

switch opts.Normalization
    case "improved"
        fv = reshape(V, 1, []);
        fv = sign(fv) .* sqrt(abs(fv));
    case "intra"
        V = V ./ max(vecnorm(V, 2, 1), eps);
        fv = reshape(V, 1, []);
    case "none"
        fv = reshape(V, 1, []);
        return
end
fv = fv / max(norm(fv), eps);
end
