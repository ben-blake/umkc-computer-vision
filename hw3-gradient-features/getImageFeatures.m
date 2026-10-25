function [f, loc] = getImageFeatures(im, opt)
%getImageFeatures - Local gradient features of an image
%   f = getImageFeatures(im, opt) converts the image im to grayscale and
%   returns its local gradient features as an n-by-d matrix, one feature
%   per row. opt is a structure whose field type selects the feature:
%       "sift" - (default) SIFT. Key points from detectSIFTFeatures,
%                described by extractFeatures. d = 128, and n depends on
%                the image.
%       "dsft" - Dense SIFT. The same 4-by-4-by-8 gradient histogram as
%                SIFT, computed upright on a regular grid instead of at
%                detected key points, the way vl_dsift does it. d = 128.
%       "hogf" - HoG. The Dalal-Triggs blocks of extractHOGFeatures, each
%                2-by-2 cells of 9 orientation bins, taken as separate
%                local features instead of one long vector. d = 36.
%
%   SIFT and dense SIFT descriptors are on the 0 to 255 scale that
%   vl_sift uses, so the two can be compared. HoG blocks are L2-hys
%   normalized as extractHOGFeatures returns them.
%
%   The structure can also hold these fields, which take their defaults
%   when they are missing:
%       step     - dense SIFT grid spacing in pixels. The default is 8.
%       binSize  - dense SIFT spatial bin width in pixels, so the
%                  descriptor covers 4*binSize. The default is 8.
%       cellSize - HoG cell width in pixels. The default is 8.
%
%   [f, loc] = getImageFeatures(...) also returns the n-by-2 [x y]
%   location of every feature in pixels.
%
%   Example: Dense SIFT of one image
%       im = imread("NWPU-RESISC45/beach/beach_001.jpg");
%       f = getImageFeatures(im, struct("type", "dsft"));
%       size(f)
%
%   See also detectSIFTFeatures, extractFeatures, extractHOGFeatures,
%   getFisherVectorModel

%   Homework 3, Q1 - gradient features.

arguments
    im {mustBeNumeric, mustBeNonempty}
    opt (1,1) struct = struct("type", "sift")
end

opt = iDefaults(opt);
gray = im2gray(im);

switch opt.type
    case "sift"
        points = detectSIFTFeatures(gray);
        [f, valid] = extractFeatures(gray, points);
        f = single(f);
        loc = double(valid.Location);
    case "dsft"
        [f, loc] = iDenseSIFT(im2double(gray), opt.step, opt.binSize);
    case "hogf"
        [f, loc] = iHOGBlocks(gray, opt.cellSize);
end
end

function opt = iDefaults(opt)
%iDefaults - Fill in missing fields and check the type
defaults = struct("type", "sift", "step", 8, "binSize", 8, "cellSize", 8);
for name = string(fieldnames(defaults))'
    if ~isfield(opt, name)
        opt.(name) = defaults.(name);
    end
end
opt.type = string(opt.type);
if ~ismember(opt.type, ["sift" "dsft" "hogf"])
    error("getImageFeatures:badType", ...
        "opt.type must be ""sift"", ""dsft"" or ""hogf"", not ""%s"".", opt.type);
end
end

function [f, loc] = iDenseSIFT(gray, step, binSize)
%iDenseSIFT - Upright SIFT descriptors on a regular grid
%   Follows vl_dsift. The gradient orientation is split between the two
%   nearest of 8 orientation bins, which gives 8 gradient images. Each is
%   pooled with a triangular window one bin wide, so that every pixel
%   votes into the spatial bins around it in proportion to how close it
%   is, and the pooled images are then read at the 16 bin centers of each
%   grid point.
nOri = 8;
nBin = 4;
[H, W] = size(gray);

% vl_phow smooths by binSize / 6 before taking the gradient.
smoothed = imgaussfilt(gray, binSize / 6);
[gx, gy] = imgradientxy(smoothed, "central");
mag = hypot(gx, gy);
t = mod(atan2(gy, gx), 2*pi) / (2*pi) * nOri;
binLo = floor(t);
w = t - binLo;
binLo = mod(binLo, nOri) + 1;
binHi = mod(binLo, nOri) + 1;

pooled = zeros(H, W, nOri);
tri = [1:binSize, binSize-1:-1:1] / binSize;
for o = 1:nOri
    vote = mag .* ((binLo == o) .* (1 - w) + (binHi == o) .* w);
    pooled(:, :, o) = conv2(tri, tri, vote, "same");
end

% Bin centers sit 0.5 and 1.5 bins from the grid point, and each pools
% binSize - 1 pixels further out, so the first grid point is 2.5 bins in.
offs = ((1:nBin) - (nBin + 1)/2) * binSize;
margin = max(offs) + binSize - 1;
xs = (1 + margin):step:(W - margin);
ys = (1 + margin):step:(H - margin);
[gx0, gy0] = meshgrid(xs, ys);
loc = [gx0(:) gy0(:)];

% Orientation varies fastest, then x, then y, as in vl_dsift.
f = zeros(numel(gx0), nBin*nBin*nOri);
for iy = 1:nBin
    for ix = 1:nBin
        block = pooled(ys + offs(iy), xs + offs(ix), :);
        col = ((iy-1)*nBin + (ix-1)) * nOri + (1:nOri);
        f(:, col) = reshape(block, [], nOri);
    end
end

% SIFT normalization: unit length, clip at 0.2 to limit the weight of
% any one large gradient, unit length again, then scale to 0 to 255.
f = f ./ max(vecnorm(f, 2, 2), eps);
f = min(f, 0.2);
f = f ./ max(vecnorm(f, 2, 2), eps);
f = single(min(round(512 * f), 255));
end

function [f, loc] = iHOGBlocks(gray, cellSize)
%iHOGBlocks - HoG blocks as separate 36-dimensional local features
%   extractHOGFeatures concatenates the blocks column by column, each block
%   holding its 2-by-2 cells of 9 bins together, so a reshape recovers one
%   row per block.
[h, ~] = extractHOGFeatures(gray, CellSize=[cellSize cellSize], ...
    BlockSize=[2 2], BlockOverlap=[1 1], NumBins=9);
d = 2*2*9;
f = single(reshape(h, d, [])');

[H, W] = size(gray);
nY = floor(H / cellSize) - 1;
nX = floor(W / cellSize) - 1;
[bx, by] = meshgrid(1:nX, 1:nY);
loc = [bx(:) by(:)] * cellSize + 0.5;   % shared corner of the 4 cells
end
