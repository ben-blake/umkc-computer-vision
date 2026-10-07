function [log_pyramid] = getLoGPyramid(im, scales)
%getLoGPyramid - Scale-normalized Laplacian of Gaussian pyramid
%   log_pyramid = getLoGPyramid(im, scales) filters the image im with a
%   Laplacian of Gaussian at each of the N standard deviations in scales
%   and returns the responses as an H-by-W-by-N array, where page k is the
%   response at scales(k). All pages keep the size of the input, so this
%   is a stack over scale and not a subsampled pyramid.
%
%   Each page is multiplied by scales(k)^2. The raw LoG response falls off
%   as 1/sigma^2, so without this factor the smallest scale would always
%   win and responses could not be compared across pages.
%
%   The response is signed. It is negative at the center of a bright blob
%   and positive at the center of a dark one. Take abs to get the strength
%   of the response regardless of polarity.
%
%   im can be RGB or grayscale of any numeric class. It is converted to
%   grayscale and to double on the 0 to 255 range, so a response has the
%   same units as an 8-bit intensity difference.
%
%   Example: Pyramid of the seven scales in the assignment
%       im = imread("flarsheim-1.jpg");
%       scales = [1.200 1.518 1.920 2.429 3.072 3.886 4.915];
%       P = getLoGPyramid(im, scales);
%       imagesc(abs(P(:,:,3))); axis image; colormap gray
%
%   See also fspecial, imfilter, getScaleResponseExtrema, getLoGKeyPoints

%   Homework 2, Q1 - LoG pyramid.

arguments
    im {mustBeNumeric, mustBeNonempty}
    scales (1,:) double {mustBePositive}
end

gray = 255 * im2double(im2gray(im));

[H, W] = size(gray);
N = numel(scales);
log_pyramid = zeros(H, W, N);
for k = 1:N
    sigma = scales(k);
    % The kernel reaches 4 sigma each side. At that radius the LoG has
    % decayed to under 0.5% of its peak, so truncation does not matter.
    halfWidth = ceil(4*sigma);
    kernel = fspecial("log", 2*halfWidth + 1, sigma);
    response = imfilter(gray, kernel, "replicate", "conv");
    log_pyramid(:, :, k) = sigma^2 * response;
end
end
