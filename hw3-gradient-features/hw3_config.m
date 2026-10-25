function cfg = hw3_config()
%hw3_config - Shared settings for the Homework 3 scripts
%   cfg = hw3_config() returns a structure with the settings every
%   Homework 3 script uses, so that the features, the models, the figures
%   and the checks all describe the same experiment. The fields are:
%
%     files        - the 600 image files, 40 from each of the first 15
%                    classes of NWPU-RESISC45, ordered class by class
%     labels       - their integer class labels, 1 to 15
%     classes      - the 15 class names as a string column
%     types        - the three feature types passed to getImageFeatures
%     typeNames    - how each type is named in tables and figures
%     kdList       - PCA dimensions of the GMM models
%     ncList       - numbers of GMM components
%     perImage     - descriptors sampled from each image to train the
%                    PCA and the GMMs
%     seed         - random seed for that sample and for the GMM fit
%     resultsDir   - folder the figures and numbers are written to
%
%   Example: Count the images of each class
%       cfg = hw3_config();
%       disp(accumarray(cfg.labels, 1)')
%
%   See also getImageFeatures, getFisherVectorModel, hw3_q1_features

%   Homework 3 - shared experiment definition.

here = fileparts(mfilename("fullpath"));
dataRoot = fullfile(here, "NWPU-RESISC45");
nImg = 40;

cfg = struct();
cfg.classes = ["airplane"; "airport"; "baseball_diamond"; ...
    "basketball_court"; "beach"; "bridge"; "chaparral"; "church"; ...
    "circular_farmland"; "cloud"; "commercial_area"; ...
    "dense_residential"; "desert"; "forest"; "freeway"];

nClass = numel(cfg.classes);
cfg.files = strings(nClass*nImg, 1);
cfg.labels = zeros(nClass*nImg, 1);
for k = 1:nClass
    d = dir(fullfile(dataRoot, cfg.classes(k), "*.jpg"));
    if numel(d) < nImg
        error("hw3_config:tooFewImages", ...
            "Class %s has %d images, %d needed.", ...
            cfg.classes(k), numel(d), nImg);
    end
    names = sort(string({d.name}'));    % airplane_001.jpg first
    offs = (k-1)*nImg;
    cfg.files(offs + (1:nImg)) = fullfile(dataRoot, cfg.classes(k), names(1:nImg));
    cfg.labels(offs + (1:nImg)) = k;
end

cfg.types = ["sift" "dsft" "hogf"];
cfg.typeNames = ["SIFT" "Dense SIFT" "HoG"];
cfg.kdList = [16 32];
cfg.ncList = [32 64 128];
cfg.perImage = 100;
cfg.seed = 0;
cfg.resultsDir = fullfile(here, "results");
end
