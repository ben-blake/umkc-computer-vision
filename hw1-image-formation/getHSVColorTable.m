function [color_table] = getHSVColorTable(im_path, K, opts)
%   getHSVColorTable - Learn an HSV color code book with k-means
%   color_table = getHSVColorTable(im_path, K) samples pixels from a set
%   of RGB images, converts them to HSV, and clusters them into K
%   centroids with k-means. im_path is either a folder that is searched
%   recursively for image files, or a string array or cell array of
%   image file names. color_table is a K-by-3 matrix whose rows are the
%   HSV centroids, with all values in [0,1] and rows sorted by hue.
%
%   color_table = getHSVColorTable(...,PixelsPerImage=NPIX) also
%   specifies how many pixels are sampled at random from each image.
%   The default is 500, which keeps k-means tractable while still
%   covering the color range of the data set.
%
%   color_table = getHSVColorTable(...,Replicates=R) also specifies the
%   number of k-means restarts. The default is 3.
%
%   color_table = getHSVColorTable(...,Seed=S) also specifies the seed
%   of the random number generator used for pixel sampling and k-means
%   initialization. The default is 0, so the code book is reproducible.
%
%   Example: Build a 64-entry code book and show it as a color strip
%       ct = getHSVColorTable("NWPU-RESISC45/beach", 64, PixelsPerImage=100);
%       image(reshape(hsv2rgb(ct), [1 64 3]))
%
%   See also getHSVHist, kmeans, rgb2hsv, imageDatastore
 
%   Homework 1, Part 1 - HSV color code book.
 
arguments
    im_path
    K (1,1) double {mustBeInteger, mustBePositive}
    opts.PixelsPerImage (1,1) double {mustBeInteger, mustBePositive} = 500
    opts.Replicates (1,1) double {mustBeInteger, mustBePositive} = 3
    opts.Seed (1,1) double = 0
end
 
files = iResolveFiles(im_path);
nFile = numel(files);
if nFile == 0
    error("getHSVColorTable:noImages", "No image files found.");
end
 
% Sample a fixed number of pixels per image so that every image
% contributes equally to the code book, independent of its size.
rng(opts.Seed);
nPix = opts.PixelsPerImage;
X = zeros(nFile*nPix, 3);
for i = 1:nFile
    im = rgb2hsv(imread(files(i)));
    [ht, wid, ~] = size(im);
    px = reshape(im, [ht*wid, 3]);
    sel = randi(ht*wid, [nPix, 1]);
    X((i-1)*nPix + (1:nPix), :) = px(sel, :);
end
 
% Cluster the sampled pixels. Squared Euclidean distance treats hue as
% a linear coordinate, so the wrap-around at H = 0 and H = 1 splits red
% into two clusters. This is fine here because getHSVHist uses the same
% geometry when it assigns pixels to bins.
kmOpts = statset("MaxIter", 300);
[~, C] = kmeans(X, K, Distance="sqeuclidean", Start="plus", ...
    Replicates=opts.Replicates, Options=kmOpts, Display="off");
 
% Sort by hue, then saturation, then value so that neighboring bins
% hold visually similar colors and the histograms are readable.
color_table = sortrows(C, [1 2 3]);
end
 
function files = iResolveFiles(im_path)
%iResolveFiles - Turn a folder or a file list into a string array
if isstring(im_path) || ischar(im_path) || iscellstr(im_path)
    im_path = string(im_path);
else
    error("getHSVColorTable:badPath", ...
        "im_path must be a folder name or a list of file names.");
end
 
if isscalar(im_path) && isfolder(im_path)
    ds = imageDatastore(im_path, IncludeSubfolders=true);
    files = string(ds.Files);
else
    files = im_path(:);
end
end
