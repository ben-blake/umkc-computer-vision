function [files, labels, classes] = hw1_config(nImg)
%hw1_config - File list and labels of the 15-class NWPU subset
%   [FILES, LABELS] = hw1_config() returns the 600 image files of the
%   homework data set, 40 images from each of the first 15 classes of
%   NWPU-RESISC45, together with their integer class labels 1 to 15.
%   FILES is a 600-by-1 string array ordered class by class, so class K
%   occupies rows (K-1)*40+1 through K*40.
%
%   [FILES, LABELS] = hw1_config(NIMG) also specifies how many images
%   are taken from each class. The default is 40.
%
%   [FILES, LABELS, CLASSES] = hw1_config(...) also returns the class
%   names as a 15-by-1 string array.
%
%   Example: Show the class of the 100th image
%       [files, labels, classes] = hw1_config();
%       disp(classes(labels(100)))
%
%   See also getHSVColorTable, getHSVHist, imageDatastore
 
%   Homework 1 - shared data set definition.
 
arguments
    nImg (1,1) double {mustBeInteger, mustBePositive} = 40
end
 
dataRoot = fullfile(fileparts(mfilename("fullpath")), "NWPU-RESISC45");
classes = ["airplane"; "airport"; "baseball_diamond"; ...
    "basketball_court"; "beach"; "bridge"; "chaparral"; "church"; ...
    "circular_farmland"; "cloud"; "commercial_area"; ...
    "dense_residential"; "desert"; "forest"; "freeway"];
 
nClass = numel(classes);
files = strings(nClass*nImg, 1);
labels = zeros(nClass*nImg, 1);
for k = 1:nClass
    d = dir(fullfile(dataRoot, classes(k), "*.jpg"));
    if numel(d) < nImg
        error("hw1_config:tooFewImages", ...
            "Class %s has %d images, %d requested.", ...
            classes(k), numel(d), nImg);
    end
    names = sort(string({d.name}'));    % airplane_001.jpg first
    offs = (k-1)*nImg;
    files(offs + (1:nImg)) = fullfile(dataRoot, classes(k), names(1:nImg));
    labels(offs + (1:nImg)) = k;
end
end
