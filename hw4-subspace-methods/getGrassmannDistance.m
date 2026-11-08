function [d, theta] = getGrassmannDistance(A, B)
%getGrassmannDistance - Cauchy-Binet distance between two subspaces
%   d = getGrassmannDistance(A, B) returns the Cauchy-Binet distance
%   between the subspaces spanned by the columns of the D-by-p matrix A
%   and the D-by-q matrix B,
%       d = sqrt(1 - prod(cos(theta).^2))
%   where theta are the min(p, q) principal angles between them. d is 0
%   when one subspace contains the other and 1 when some direction of
%   one is orthogonal to all of the other. A and B do not need
%   orthonormal columns; they are orthonormalized first.
%
%   [d, theta] = getGrassmannDistance(...) also returns the principal
%   angles in radians, smallest first.
%
%   Because it multiplies min(p, q) cosines, the distance grows quickly
%   with the dimension: sixteen angles of 30 degrees already give 0.99.
%
%   Example: Two planes in 3-D that share one direction
%       A = [1 0; 0 1; 0 0];
%       B = [1 0; 0 0; 0 1];
%       [d, theta] = getGrassmannDistance(A, B)   % d = 1, theta = [0; pi/2]
%
%   See also subspace, orth, svd, hw4_q4_local

%   Homework 4, Q4 - Grassmann distance.

arguments
    A (:,:) double
    B (:,:) double
end

if size(A, 1) ~= size(B, 1)
    error("getGrassmannDistance:sizeMismatch", ...
        "A and B must have the same number of rows.");
end

Qa = orth(A);
Qb = orth(B);
% The singular values of Qa'*Qb are the cosines of the principal angles.
c = svd(Qa' * Qb);
c = min(c(1:min(size(Qa, 2), size(Qb, 2))), 1);
theta = acos(c);
d = sqrt(max(0, 1 - prod(c.^2)));
end
