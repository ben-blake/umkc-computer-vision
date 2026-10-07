function [x, y, scale, strength] = getLoGKeyPoints(im, log_peak_thres, opts)
%getLoGKeyPoints - Key points at scale space extrema of a LoG pyramid
%   [x, y, scale] = getLoGKeyPoints(im, log_peak_thres) detects key points
%   in the image im as extrema of the scale-normalized LoG response over
%   both position and scale. x and y are the column and row of each key
%   point in pixels, and scale is the standard deviation at which its
%   response peaks. The outputs are M-by-1 and sorted from the strongest
%   key point to the weakest.
%
%   A pixel becomes a key point when it passes three tests:
%
%     1. Strength. The magnitude of its LoG response, taken at whichever
%        scale is largest for that pixel, is at least log_peak_thres.
%     2. Position. That magnitude is larger than its 26 neighbors in the
%        3-by-3-by-3 block spanning the pixels around it on the same
%        pyramid level and on the levels directly above and below.
%     3. Scale. A cubic fitted to its responses across all scales has a
%        maximum strictly inside the scale range, the fitted peak is also
%        at least log_peak_thres, and the fitted scale is no further than
%        one pyramid level from the level tested in 2. See
%        getScaleResponseExtrema.
%
%   log_peak_thres is in gray levels on the 0 to 255 range.
%
%   [x, y, scale, strength] = getLoGKeyPoints(...) also returns the fitted
%   peak response of each key point.
%
%   [...] = getLoGKeyPoints(..., Scales=S) also specifies the scales of
%   the pyramid. The default is the seven scales given in the assignment,
%   [1.200 1.518 1.920 2.429 3.072 3.886 4.915].
%
%   Example: Detect and plot key points
%       im = imread("flarsheim-1.jpg");
%       [x, y, scale] = getLoGKeyPoints(im, 16);
%       imshow(im); hold on; plot(x, y, "r+")
%
%   See also getLoGPyramid, getScaleResponseExtrema, imregionalmax,
%   detectSIFTFeatures

%   Homework 2, Q3 - image level LoG key points.

arguments
    im {mustBeNumeric, mustBeNonempty}
    log_peak_thres (1,1) double {mustBeNonnegative}
    opts.Scales (1,:) double {mustBePositive} = ...
        [1.200 1.518 1.920 2.429 3.072 3.886 4.915]
end

scales = opts.Scales;
N = numel(scales);

% Blobs of either polarity count, so work with the magnitude.
P = abs(getLoGPyramid(im, scales));
[H, W, ~] = size(P);

% Tests 1 and 2: strong enough, and larger than all 26 neighbors in the
% 3-by-3-by-3 block around it. Only the level where the pixel responds
% most is tested, so a pixel can produce at most one key point.
[peakOverScale, level] = max(P, [], 3);
isBlockMax = imregionalmax(P, 26);
[rowGrid, colGrid] = ndgrid(1:H, 1:W);
isCandidate = isBlockMax(sub2ind([H W N], rowGrid, colGrid, level));
isCandidate = isCandidate & (peakOverScale >= log_peak_thres);

% The filter has no real support at the border, so drop a 1 pixel frame.
isCandidate([1 end], :) = false;
isCandidate(:, [1 end]) = false;

[row, col] = find(isCandidate);
candidateLevel = level(isCandidate);
nCandidate = numel(row);

% Test 3: an extremum over scale, located by the cubic fit.
scale = nan(nCandidate, 1);
strength = nan(nCandidate, 1);
for i = 1:nCandidate
    response = reshape(P(row(i), col(i), :), 1, N);
    [scale(i), strength(i)] = getScaleResponseExtrema(scales, response);
end

% The fit refines the sampled maximum, so it has to agree with it. A
% response with two bumps can put the cubic's peak several levels away
% from the level whose neighborhood was tested, and that is not the same
% extremum.
lowerScale = scales(max(candidateLevel - 1, 1))';
upperScale = scales(min(candidateLevel + 1, N))';
isNearLevel = (scale >= lowerScale) & (scale <= upperScale);

isKeyPoint = isNearLevel & (strength >= log_peak_thres);
[strength, order] = sort(strength(isKeyPoint), "descend");

x = col(isKeyPoint);
y = row(isKeyPoint);
scale = scale(isKeyPoint);
x = x(order);
y = y(order);
scale = scale(order);
end
