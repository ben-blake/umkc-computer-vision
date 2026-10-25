function [fv] = getFisherVector(f, A, gmm, kd, nc, opts)
%getFisherVector - Fisher vector of a set of local features
%   fv = getFisherVector(f, A, gmm, kd, nc) aggregates the n-by-d local
%   features f of one image into a single 1-by-2*kd*nc Fisher vector. The
%   features are reduced to kd dimensions with the PCA matrix A, and
%   then described by how they pull on each of the nc components of the
%   diagonal GMM gmm, both returned by getFisherVectorModel.
%
%   For component k with weight w, mean m and standard deviation s, and
%   q(i,k) the posterior probability that feature x(i) came from it,
%       u_k = sum_i q(i,k) * (x(i) - m) ./ s           / (n*sqrt(w))
%       v_k = sum_i q(i,k) * ((x(i) - m).^2 ./ s.^2 - 1) / (n*sqrt(2*w))
%   are the gradients of the log likelihood with respect to the mean and
%   the deviation, whitened by the Fisher information. fv is
%   [u_1 v_1 u_2 v_2 ... u_nc v_nc], which is the order vl_fisher uses.
%
%   fv = getFisherVector(..., Normalization=NORM) also specifies the
%   normalization applied to the result. NORM must be:
%       "improved" - (default) signed square root of every entry, then
%                    unit L2 norm. This is the improved Fisher vector of
%                    Perronnin et al. and vl_fisher's "Improved" option.
%       "none"     - the raw gradients above.
%
%   An image with no features has a Fisher vector of zeros.
%
%   Example: Fisher vector of one image
%       f = double(getImageFeatures(imread("NWPU-RESISC45/beach/beach_001.jpg")));
%       [gmm, A] = getFisherVectorModel(f, 16, 32);
%       fv = getFisherVector(f, A, gmm, 16, 32);
%       numel(fv)            % 2*16*32 = 1024
%
%   See also getFisherVectorModel, getVLAD, vl_fisher

%   Homework 3, Q3 - Fisher vector aggregation.

arguments
    f (:,:) {mustBeNumeric}
    A (:,:) double
    gmm (1,1) struct
    kd (1,1) double {mustBeInteger, mustBePositive}
    nc (1,1) double {mustBeInteger, mustBePositive}
    opts.Normalization (1,1) string {mustBeMember(opts.Normalization, ...
        ["improved", "none"])} = "improved"
end

if ~isequal(size(gmm.mean), [kd nc]) || ~isequal(size(gmm.var), [kd nc])
    error("getFisherVector:gmmSize", ...
        "gmm.mean and gmm.var must be %d-by-%d.", kd, nc);
end

n = size(f, 1);
if n == 0
    fv = zeros(1, 2*kd*nc);
    return
end
x = double(f) * A(:, 1:kd);                 % n-by-kd

mu = gmm.mean;                              % kd-by-nc
s2 = gmm.var;
w = gmm.prior(:)';                          % 1-by-nc

% Log of w_k * N(x_i; mu_k, s2_k), expanded so that it is three matrix
% products instead of a loop over components.
logLik = -0.5 * ((x.^2) * (1 ./ s2) - 2 * x * (mu ./ s2) ...
    + sum(mu.^2 ./ s2, 1)) - 0.5 * sum(log(2*pi*s2), 1) + log(w);
% Posteriors, with the largest term taken out so exp cannot underflow.
logLik = logLik - max(logLik, [], 2);
q = exp(logLik);
q = q ./ sum(q, 2);                         % n-by-nc

% Zeroth, first and second order statistics of each component.
S0 = sum(q, 1);                             % 1-by-nc
S1 = x' * q;                                % kd-by-nc
S2 = (x.^2)' * q;                           % kd-by-nc

u = (S1 - mu .* S0) ./ sqrt(s2) ./ (n * sqrt(w));
v = ((S2 - 2 * mu .* S1 + mu.^2 .* S0) ./ s2 - S0) ./ (n * sqrt(2*w));
fv = reshape([u; v], 1, []);

if opts.Normalization == "improved"
    fv = sign(fv) .* sqrt(abs(fv));
    fv = fv / max(norm(fv), eps);
end
end
