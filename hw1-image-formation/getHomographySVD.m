function [H, sv, h, A] = getHomographySVD(p1, p2, opts)
%getHomographySVD - Homography from point correspondences by SVD
%   H = getHomographySVD(p1, p2) solves the homogeneous system A*h = 0 for
%   the 3-by-3 projective transform that maps the M-by-2 points p1 onto the
%   M-by-2 points p2. Each correspondence contributes two rows to A, so A
%   is 2M-by-9 and the system is over-determined once M is more than four.
%   The solution is the right singular vector of A belonging to the
%   smallest singular value, reshaped to 3-by-3 and scaled so H(3,3) = 1.
%
%   [H, sv] = getHomographySVD(...) also returns the nine singular values
%   of A in descending order.
%
%   [H, sv, h] = getHomographySVD(...) also returns the 9-by-1 null space
%   vector before it is reshaped into a matrix.
%
%   [H, sv, h, A] = getHomographySVD(...) also returns the matrix A itself.
%
%   [...] = getHomographySVD(..., Normalize=TF) also specifies whether the
%   points are conditioned by the Hartley transform before A is built, with
%   the result mapped back afterwards. The default is false, which solves
%   the system exactly as it is written in the assignment.
%
%   Example: Homography between four corners and a unit square
%       src = [0 0; 1 0; 1 1; 0 1];
%       dst = [0 0; 2 0; 2 2; 0 2];
%       H = getHomographySVD(src, dst)
%
%   See also svd, maketform, imtransform, matchFeatures
 
%   Homework 1, Part 2 - direct linear transform solved by SVD.
 
arguments
    p1 (:,2) double
    p2 (:,2) double
    opts.Normalize (1,1) logical = false
end
 
M = size(p1, 1);
if size(p2, 1) ~= M
    error("getHomographySVD:sizeMismatch", ...
        "p1 and p2 must have the same number of rows.");
end
if M < 4
    error("getHomographySVD:tooFewPoints", ...
        "At least 4 correspondences are needed, %d given.", M);
end
 
if opts.Normalize
    [q1, T1] = iNormalize(p1);
    [q2, T2] = iNormalize(p2);
else
    q1 = p1; q2 = p2;
    T1 = eye(3); T2 = eye(3);
end
 
% Two rows per correspondence. Writing x' = (h11*x + h12*y + h13) / w and
% y' = (h21*x + h22*y + h23) / w with w = h31*x + h32*y + h33, then
% clearing the denominator, gives the two linear constraints below.
A = zeros(2*M, 9);
for i = 1:M
    x  = q1(i,1);  y  = q1(i,2);
    xp = q2(i,1);  yp = q2(i,2);
    A(2*i-1, :) = [x, y, 1, 0, 0, 0, -xp*x, -xp*y, -xp];
    A(2*i,   :) = [0, 0, 0, x, y, 1, -yp*x, -yp*y, -yp];
end
 
% The null space of A is the right singular vector with the smallest
% singular value, which is the last column of V.
[~, S, V] = svd(A);
sv = diag(S);
h = V(:, end);
H = reshape(h, [3 3])';
 
if opts.Normalize
    H = T2 \ H * T1;
end
if H(3,3) ~= 0
    H = H / H(3,3);
end
end
 
function [q, T] = iNormalize(p)
%iNormalize - Hartley conditioning: centroid at origin, mean distance sqrt(2)
c = mean(p, 1);
d = mean(vecnorm(p - c, 2, 2));
s = sqrt(2) / d;
T = [s 0 -s*c(1); 0 s -s*c(2); 0 0 1];
q = (T * [p, ones(size(p,1), 1)]')';
q = q(:, 1:2);
end
