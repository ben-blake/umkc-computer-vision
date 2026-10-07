function cfg = hw2_config()
%hw2_config - Shared settings for the Homework 2 scripts
%   cfg = hw2_config() returns a structure with the settings every
%   Homework 2 script uses, so that the figures, the saved results and the
%   checks all describe the same experiment. The fields are:
%
%     scales       - the seven LoG standard deviations in the assignment
%     images       - full paths of the test images, as a string column
%     logPeakThres - LoG peak threshold in gray levels on 0 to 255
%     thresSweep   - thresholds used for the count versus threshold table
%     budgets      - key point budgets for the equal-count comparison
%     matchRadius  - distance in pixels within which two key points from
%                    different detectors are called the same point
%     edgeRatio    - principal curvature ratio of the SIFT edge test
%     resultsDir   - folder the figures and numbers are written to
%
%   Example: Show the scale step between pyramid levels
%       cfg = hw2_config();
%       disp(cfg.scales(2:end) ./ cfg.scales(1:end-1))
%
%   See also getLoGPyramid, getLoGKeyPoints, hw2_q3_keypoints

%   Homework 2 - shared experiment definition.

here = fileparts(mfilename("fullpath"));

cfg = struct();
cfg.scales = [1.200 1.518 1.920 2.429 3.072 3.886 4.915];
cfg.images = fullfile(here, ["flarsheim-1.jpg"; "sift-1.jpg"; "sift-3.png"]);
cfg.logPeakThres = 16;
cfg.thresSweep = [8 16 24 32 40 50 60];
cfg.budgets = [100 300 1000];
cfg.matchRadius = 3;
cfg.edgeRatio = 10;
cfg.resultsDir = fullfile(here, "results");
end
