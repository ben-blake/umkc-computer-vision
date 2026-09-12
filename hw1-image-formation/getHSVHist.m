function [h] = getHSVHist(im, color_table, opt)
%getHSVHist - Color histogram of an image over an HSV code book
%   H = getHSVHist(IM, color_table) converts the RGB image IM to HSV,
%   assigns every pixel to the closest entry of color_table, and returns
%   the normalized 1-by-K histogram of those assignments. color_table is
%   the K-by-3 code book returned by getHSVColorTable.
%
%   H = getHSVHist(IM, color_table, OPT) also specifies the metric used
%   to match pixels to code book entries. OPT must be:
%       "euc" - (default) Euclidean distance in HSV space, computed
%               with pdist2.
%       "kld" - Kullback-Leibler divergence. The pixel and the code book
%               entry are each normalized to sum to one and compared as
%               3-bin distributions over the H, S, and V channels.
%
%   The histogram is divided by the number of pixels, so H sums to one
%   and images of different sizes stay comparable.
%
%   Example: Histogram of one image over a 64-entry code book
%       ct = getHSVColorTable("NWPU-RESISC45/beach", 64, PixelsPerImage=100);
%       h = getHSVHist(imread("NWPU-RESISC45/beach/beach_001.jpg"), ct);
%       bar(h)
%
%   See also getHSVColorTable, pdist2, rgb2hsv, histogram
 
%   Homework 1, Part 2 - HSV color histogram feature.
 
arguments
    im
    color_table (:,3) double
    opt (1,1) string {mustBeMember(opt, ["euc", "kld"])} = "euc"
end
 
im = rgb2hsv(im);
[ht, wid, ~] = size(im);
K = size(color_table, 1);
X = reshape(im, [ht*wid, 3]);
 
if opt == "euc"                 % euclidean distance
    dist = pdist2(color_table, X);
elseif opt == "kld"             % KL distance (cross entropy)
    dist = iKLDivergence(color_table, X);
end
 
% Vector quantization: each pixel votes for its closest code word.
[~, idx] = min(dist, [], 1);
h = accumarray(idx(:), 1, [K 1])' / (ht*wid);
end
 
function D = iKLDivergence(C, X)
%iKLDivergence - KL divergence from pixels to code words, K-by-M
%   Row K, column M of D is KL(x_m || c_k). The channels of each pixel
%   and each code word are treated as a 3-bin distribution, so both are
%   normalized to sum to one. A small offset keeps log(0) out of the
%   computation and gives black pixels, whose channels sum to zero, a
%   well defined distribution.
a = 1e-6;
P = (X + a) ./ sum(X + a, 2);           % M-by-3, pixels
Q = (C + a) ./ sum(C + a, 2);           % K-by-3, code words
 
% KL(p||q) = sum(p.*log(p)) - sum(p.*log(q)). The first term is the
% negative entropy of the pixel and is the same for every code word, so
% min() picks the code word that cross entropy would pick.
D = (sum(P .* log(P), 2) - P * log(Q)')';
end
