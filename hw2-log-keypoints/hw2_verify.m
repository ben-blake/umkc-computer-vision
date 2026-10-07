%hw2_verify - Independent checks on the three Homework 2 functions
%   Tests getLoGPyramid, getScaleResponseExtrema and getLoGKeyPoints
%   against cases whose answer is known in advance: a LoG kernel written
%   out from its formula, cubics with a known maximum, and synthetic
%   Gaussian blobs of known position, size and contrast. Then confirms
%   that the saved results reproduce. Prints PASS or FAIL for each check.
%
%   See also getLoGPyramid, getScaleResponseExtrema, getLoGKeyPoints

%   Homework 2 - verification harness.

clear; clc; rng(0);
chk = @(tf, name) fprintf("%-66s %s\n", name, ...
    string(tf).replace("true","PASS").replace("false","FAIL"));
cfg = hw2_config();
scales = cfg.scales;
N = numel(scales);
im = imread(cfg.images(1));
gray = 255 * im2double(im2gray(im));

fprintf("--- Q1: getLoGPyramid ---\n");
P = getLoGPyramid(im, scales);
chk(isequal(size(P), [size(gray) N]), "pyramid is H by W by 7");
chk(isa(P, "double") && all(isfinite(P(:))), "pyramid is finite double");

% The LoG kernel from its definition, without fspecial. The mean is
% removed so that the kernel sums to zero, as a Laplacian must.
k = 4; sigma = scales(k); hw = ceil(4*sigma);
[u, v] = meshgrid(-hw:hw);
g = exp(-(u.^2 + v.^2) / (2*sigma^2));
kernel = (u.^2 + v.^2 - 2*sigma^2) .* g / (sigma^4 * sum(g(:)));
kernel = kernel - mean(kernel(:));
direct = sigma^2 * conv2(padarray(gray, [hw hw], "replicate"), kernel, "valid");
chk(max(abs(direct - P(:,:,k)), [], "all") < 1e-9, ...
    "level 4 equals sigma^2 times the kernel written from the formula");

flat = getLoGPyramid(100 * ones(60, 80), scales);
chk(max(abs(flat(:))) < 1e-9, "constant image gives zero response");
ramp = getLoGPyramid(repmat(1:80, 60, 1), scales);
chk(max(abs(ramp(20:40, 30:50, :)), [], "all") < 1e-9, ...
    "linear ramp gives zero response away from the border");
P2 = getLoGPyramid(uint8(gray / 2), scales);
Phalf = getLoGPyramid(double(uint8(gray / 2)) / 255, scales);
chk(max(abs(P2 - Phalf), [], "all") < 1e-9, ...
    "uint8 and double inputs of the same image agree");
chk(max(abs(getLoGPyramid(im, scales) - getLoGPyramid(im2gray(im), scales)), ...
    [], "all") < 1e-9, "RGB input equals its grayscale conversion");

% A Gaussian blob of standard deviation s and height A. In theory the
% scale-normalized LoG at its center is -2*A*s^2*sigma^2/(s^2+sigma^2)^2,
% which is largest in magnitude at sigma = s, where it equals -A/2.
s = 2.429; A = 120; c = [101 101];
[X, Y] = meshgrid(1:201);
blob = (40 + A * exp(-((X - c(1)).^2 + (Y - c(2)).^2) / (2*s^2))) / 255;
B = getLoGPyramid(blob, scales);
center = reshape(B(c(2), c(1), :), 1, N);
theory = -2*A * s^2 * scales.^2 ./ (s^2 + scales.^2).^2;
chk(all(center < 0), "bright blob gives a negative response at its center");
chk(max(abs(center - theory) ./ abs(theory)) < 0.02, ...
    "blob center matches the closed form within 2% on every level");
[~, kMax] = max(abs(center));
chk(scales(kMax) == s, "blob response is largest on the level where sigma = s");
chk(abs(abs(center(kMax)) - A/2) < 0.02 * A/2, ...
    "and its size there is A/2, so sigma^2 normalizes correctly");
Bdark = getLoGPyramid(1 - blob, scales);
chk(max(abs(Bdark + B), [], "all") < 1e-9, ...
    "inverting the image flips the sign and nothing else");

% SIFT's difference of Gaussians approximates the normalized LoG. Since
% dG/dsigma = sigma * LoG, integrating from sigma to k*sigma gives
% DoG = ln(k) * sigma^2 LoG at the geometric mean scale. Lowe's (k-1)
% is the first order form of the same factor.
ratio = scales(2) / scales(1);
dog = imgaussfilt(gray, scales(2), FilterSize=2*ceil(4*scales(2))+1) ...
    - imgaussfilt(gray, scales(1), FilterSize=2*ceil(4*scales(2))+1);
mid = getLoGPyramid(im, sqrt(scales(1) * scales(2)));
rho = corr(dog(:), mid(:));
slope = mid(:) \ dog(:);
chk(rho > 0.99, sprintf("DoG correlates with LoG at the mid scale (r = %.4f)", rho));
chk(abs(slope / log(ratio) - 1) < 0.02, ...
    sprintf("DoG is ln(k) times the normalized LoG (slope %.3f, ln k = %.3f)", ...
    slope, log(ratio)));

fprintf("\n--- Q2: getScaleResponseExtrema ---\n");
% A cubic with a known maximum at x = 2.5 and a minimum at x = 6.
p = polyint([1 -8.5 15]);           % f' = (x - 2.5)(x - 6)
p(end) = 30;
[x0, yMax] = getScaleResponseExtrema(scales, polyval(p, scales));
chk(abs(x0 - 2.5) < 1e-9, "exact cubic: peak scale recovered to 1e-9");
chk(abs(yMax - polyval(p, 2.5)) < 1e-9, "exact cubic: peak value recovered to 1e-9");
chk(abs(polyval(polyder(p), x0)) < 1e-9, "gradient is zero at the returned scale");
chk(polyval(polyder(polyder(p)), x0) < 0, "second derivative is negative there");

x0 = getScaleResponseExtrema(scales, 50 - 8*scales);
chk(isnan(x0), "falling line: no extremum");
x0 = getScaleResponseExtrema(scales, 5 + 8*scales);
chk(isnan(x0), "rising line: no extremum");
x0 = getScaleResponseExtrema(scales, (scales - 3).^2 + 4);
chk(isnan(x0), "valley: the only critical point is a minimum, rejected");
[x0, yMax] = getScaleResponseExtrema(scales, 40 - (scales - 3).^2);
chk(abs(x0 - 3) < 1e-9 && abs(yMax - 40) < 1e-9, "parabola: vertex recovered");

% A bump that is lower than the curve at one end is not the peak.
pBump = polyint(-[1 -7.4 13.2]);     % f' = -(x - 3)(x - 4.4), max at 4.4
pBump(end) = 20;
yBump = polyval(pBump, scales);
chk(yBump(1) > polyval(pBump, 4.4), "bump test case is higher at the first scale");
chk(isnan(getScaleResponseExtrema(scales, yBump)), "bump below an end value: rejected");

% The professor's reference finds the peak by a grid search with a step
% of 0.02. Wherever both find an interior peak they must agree to a step.
% The last scale is appended because 0.02 steps from 1.2 stop at 4.90,
% short of 4.915, and the two methods then see different end values.
nTrial = 2000; nBoth = 0; worst = 0; nDisagree = 0; nNearEnd = 0;
grid = [scales(1):0.02:scales(end), scales(end)];
for t = 1:nTrial
    y = 20 + 15*randn(1, N);
    x0 = getScaleResponseExtrema(scales, y);
    pp = polyfit(scales, y, 3);
    [~, j] = max(polyval(pp, grid));
    gridInterior = j > 1 && j < numel(grid);
    if ~isnan(x0) && gridInterior
        nBoth = nBoth + 1;
        worst = max(worst, abs(x0 - grid(j)));
    end
    % The grid cannot see a peak that sits less than one step from an
    % end, so a disagreement only counts when the peak is further in.
    if isnan(x0) == gridInterior
        nearEnd = ~isnan(x0) && min(x0 - scales(1), scales(end) - x0) < 0.02;
        nDisagree = nDisagree + ~nearEnd;
        nNearEnd = nNearEnd + nearEnd;
    end
end
chk(worst <= 0.02, sprintf("matches the grid search to one step (%d cases, worst %.4f)", ...
    nBoth, worst));
chk(nDisagree == 0, sprintf(...
    "agrees with it on whether a peak exists (%d differ, %d within a step of an end)", ...
    nDisagree, nNearEnd));

chk(threw(@() getScaleResponseExtrema(scales, 1:3)), "size mismatch is an error");
chk(threw(@() getScaleResponseExtrema(1:3, 1:3)), "fewer than 4 points is an error");

fprintf("\n--- Q3: getLoGKeyPoints on synthetic blobs ---\n");
% Six blobs, bright and dark, of sizes spread across the pyramid.
centers = [60 60; 180 70; 300 60; 70 190; 190 200; 310 190];
sizes = [1.6 2.0 2.5 3.0 3.5 4.0];
heights = [120 -110 100 -120 110 -100];
[X, Y] = meshgrid(1:380, 1:260);
scene = 128 * ones(260, 380);
for i = 1:6
    scene = scene + heights(i) * exp(-((X - centers(i,1)).^2 + ...
        (Y - centers(i,2)).^2) / (2*sizes(i)^2));
end
[x, y, scale, strength] = getLoGKeyPoints(scene / 255, 30);
chk(numel(x) == 6, sprintf("six blobs give six key points (got %d)", numel(x)));
[d, j] = min(pdist2(centers, [x y]), [], 2);
chk(numel(unique(j)) == 6 && max(d) <= 1, "each lands within 1 px of a blob center");
relScale = scale(j)' ./ sizes;
chk(all(abs(relScale - 1) < 0.10), sprintf(...
    "fitted scale within 10%% of the blob size (ratios %.2f to %.2f)", ...
    min(relScale), max(relScale)));
relStrength = strength(j)' ./ (abs(heights) / 2);
chk(all(abs(relStrength - 1) < 0.05), sprintf(...
    "fitted strength within 5%% of A/2 (ratios %.2f to %.2f)", ...
    min(relStrength), max(relStrength)));
chk(issorted(strength, "descend"), "key points are sorted strongest first");

% SIFT on the same scene, for the scale convention used in the report.
sift = detectSIFTFeatures(uint8(scene));
[dS, jS] = min(pdist2(centers, double(sift.Location)), [], 2);
siftRel = double(sift.Scale(jS))' ./ sizes;
chk(max(dS) <= 1, "SIFT finds the same six centers");
fprintf("    LoG sigma / blob size:  %s\n", mat2str(relScale, 3));
fprintf("    SIFT sigma / blob size: %s\n", mat2str(siftRel, 3));
fprintf("    LoG sigma / SIFT sigma: median %.3f\n", median(relScale ./ siftRel));

% A blob too large for the pyramid keeps rising to the last level.
big = 128 + 100 * exp(-((X - 190).^2 + (Y - 130).^2) / (2*9^2));
xBig = getLoGKeyPoints(big / 255, 5);
chk(isempty(xBig), "blob larger than the scale range gives no key point");
chk(isempty(getLoGKeyPoints(128 * ones(60, 80) / 255, 1)), ...
    "constant image gives no key point");

fprintf("\n--- Q3: getLoGKeyPoints on the real image ---\n");
[x, y, scale, strength] = getLoGKeyPoints(im, cfg.logPeakThres);
chk(all(strength >= cfg.logPeakThres), "every strength is at least the threshold");
chk(all(scale > scales(1) & scale < scales(end)), ...
    "every scale is strictly inside the pyramid range");
chk(all(x > 1 & x < size(gray, 2) & y > 1 & y < size(gray, 1)), ...
    "every key point is inside the image");
chk(size(unique([x y], "rows"), 1) == numel(x), "no pixel is reported twice");

% Recompute one key point by hand from the pyramid.
i = 25;
A = abs(P);
response = reshape(A(y(i), x(i), :), 1, N);
[x0, yMax] = getScaleResponseExtrema(scales, response);
chk(abs(x0 - scale(i)) < 1e-12 && abs(yMax - strength(i)) < 1e-12, ...
    "key point 25 reproduces from the pyramid and the cubic fit");
[~, lev] = max(response);
block = A(y(i)-1:y(i)+1, x(i)-1:x(i)+1, max(lev-1,1):min(lev+1,N));
chk(A(y(i), x(i), lev) == max(block(:)) && sum(block(:) == max(block(:))) == 1, ...
    "and it is the unique maximum of its 3 by 3 by 3 block");

% A higher threshold can only remove key points.
[xHi, yHi] = getLoGKeyPoints(im, 2 * cfg.logPeakThres);
chk(all(ismember([xHi yHi], [x y], "rows")), ...
    "doubling the threshold returns a subset");
[xD, yD, sD] = getLoGKeyPoints(im, cfg.logPeakThres, Scales=scales);
chk(isequal([xD yD sD], [x y scale]), "default scales are the assignment's");
[xI, yI] = getLoGKeyPoints(255 - im, cfg.logPeakThres);
chk(isequal(sortrows([xI yI]), sortrows([x y])), ...
    "inverting the image gives the same key points");

fprintf("\n--- the reported results reproduce ---\n");
load hw2_q3_results.mat res
for n = 1:numel(res)
    [x, y, scale] = getLoGKeyPoints(imread(cfg.images(n)), cfg.logPeakThres);
    chk(isequal([x y], res(n).logXY) && isequal(scale, res(n).logScale), ...
        sprintf("%s: %d key points match the saved ones", res(n).image, numel(x)));
    sift = detectSIFTFeatures(im2gray(imread(cfg.images(n))));
    chk(sift.Count == res(n).nSift, ...
        sprintf("%s: %d SIFT key points match the saved count", res(n).image, sift.Count));
end

function tf = threw(f)
%threw - True when calling f raises an error
tf = false;
try
    f();
catch
    tf = true;
end
end
