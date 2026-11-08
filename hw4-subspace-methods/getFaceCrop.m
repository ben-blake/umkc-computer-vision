function [face, box, gray] = getFaceCrop(im, opts)
%getFaceCrop - Detect the main face in a photo and crop it to 20 by 20
%   face = getFaceCrop(im) converts the image im to grayscale, finds the
%   largest frontal face with the Viola-Jones detector, crops a square
%   around it, resizes the crop to 20-by-20 pixels and stretches it to
%   the range [0, 1], which is the form of the faces in the data set.
%   face is returned as a 1-by-400 row in column-major order, so
%   reshape(face, 20, 20) is the upright image.
%
%   [face, box] = getFaceCrop(im) also returns the square crop as
%   [x y width height] in the pixels of im.
%
%   [face, box, gray] = getFaceCrop(im) also returns the grayscale
%   image the crop was taken from.
%
%   [...] = getFaceCrop(..., Scale=S) also specifies the side of the
%   square relative to the detected box. The default is 1.0, which keeps
%   roughly brow to chin, the framing of most faces in the data set.
%
%   [...] = getFaceCrop(..., Shift=V) also moves the square down by V
%   times its side. The default is 0.05, because the detector box sits
%   slightly high on the face.
%
%   [...] = getFaceCrop(..., Size=N) also specifies the output size. The
%   default is 20.
%
%   An image in which no face is found is an error.
%
%   Example: Crop one photo and show it
%       face = getFaceCrop(imread("self/IMG_1725.PNG"));
%       imagesc(reshape(face, 20, 20)); colormap gray; axis image
%
%   See also vision.CascadeObjectDetector, imresize, hw4_q0_faces

%   Homework 4 - preparing the extra face images.

arguments
    im {mustBeNumeric, mustBeNonempty}
    opts.Scale (1,1) double {mustBePositive} = 1.0
    opts.Shift (1,1) double = 0.05
    opts.Size (1,1) double {mustBeInteger, mustBePositive} = 20
end

% im2double first: phone photos come as 16-bit PNGs as well as 8-bit JPEGs.
gray = im2gray(im2double(im));

% The detector is run under several settings and every box it returns is
% kept: copies 800, 500 and 300 pixels high, since a close-up's face can
% be larger than the detector expects; merge thresholds of 4 and 2; two
% detector models; and the photo rotated a little each way, since the
% frontal detector misses a tilted head. The largest box wins. That is
% the person the photo is about, and the false boxes a low threshold
% lets through, on a cheek or in the background, are all much smaller.
% The upright photo is preferred: a box from a rotated copy only wins
% when it is more than 1.5 times the size of the best upright box, since
% rotated copies give slightly larger boxes even for an upright head. A
% box found in a rotated photo is cropped from that rotated photo, which
% also levels the face like the ones in the data set.
angles = [0 -15 15 -30 30];
best = [];                              % [x y w h angle] in full pixels
for angle = angles
    if angle ~= 0 && ~isempty(best)
        bar = 1.5 * best(3);            % what a rotated box has to beat
    else
        bar = 0;
    end
    rotated = imrotate(gray, angle, "bilinear", "crop");
    for height = [800 500 300]
        s = min(1, height / size(gray, 1));
        small = imresize(rotated, s);
        for merge = [4 2]
            for model = ["FrontalFaceCART" "FrontalFaceLBP"]
                detector = vision.CascadeObjectDetector(model, ...
                    MergeThreshold=merge, MinSize=round([0.08 0.08] * size(small, 1)));
                boxes = detector(small) / s;
                if isempty(boxes), continue; end
                [~, k] = max(boxes(:, 3));
                if (isempty(best) || boxes(k, 3) > best(3)) && boxes(k, 3) > bar
                    best = [boxes(k, :) angle];
                end
            end
        end
    end
end
if isempty(best)
    error("getFaceCrop:noFace", "No face was found in the image.");
end
gray = imrotate(gray, best(5), "bilinear", "crop");
b = best(1:4);

side = opts.Scale * max(b(3:4));
cx = b(1) + b(3)/2;
cy = b(2) + b(4)/2 + opts.Shift * side;
box = round([cx - side/2, cy - side/2, side, side]);

% Pad by replication, so a face near the edge of the photo still gives
% a full square instead of a shifted one.
pad = ceil(side);
padded = padarray(gray, [pad pad], "replicate");
crop = imcrop(padded, box + [pad pad 0 0]);
crop = imresize(crop, [opts.Size opts.Size], "bicubic", Antialiasing=true);

% Stretch to [0, 1], as the assignment asks.
crop = crop - min(crop(:));
crop = crop / max(max(crop(:)), eps);
face = reshape(crop, 1, []);
end
