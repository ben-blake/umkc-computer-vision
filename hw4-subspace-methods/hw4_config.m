function cfg = hw4_config()
%hw4_config - Shared settings for the Homework 4 scripts
%   cfg = hw4_config() returns a structure with the settings every
%   Homework 4 script uses, so that the models, the figures, the saved
%   results and the checks all describe the same experiment. The fields
%   are:
%
%     dataFile     - the provided face data set, 6680 faces of 417 subjects
%     selfDir      - the 15 photos of me, as taken
%     celebDir     - the 10 celebrity photos and their sources
%     facesFile    - the data set augmented with the celebrity, plus my
%                    15 faces, written by hw4_q0_faces
%     selfId       - my face ID, my student ID
%     celebId      - the celebrity's face ID, one past the last subject
%     nSelfDb      - how many of my faces go into the retrieval database;
%                    the rest are the queries
%     d0           - Eigenface dimension the Fisherface and Laplacianface
%                    models are built on, as in the course code
%     d0Sweep      - Eigenface dimensions for the accuracy sweep
%     nRecog       - Q2 recognition uses the first nRecog faces
%     nRetrieve    - Q3 retrieval database uses the first nRetrieve faces
%     topK         - Q3 retrieval list length for AP
%     nLocal       - Q4 queries are drawn from the first nLocal faces
%     nQuery       - Q4 number of queries
%     kdLocal      - Q4 PCA dimension used for the KNN search
%     knnList      - Q4 neighborhood sizes
%     seed         - random seed for the Q4 query draw
%     resultsDir   - folder the figures and numbers are written to
%
%   Example: My face ID
%       cfg = hw4_config();
%       disp(cfg.selfId)
%
%   See also getEigenface, getFisherface, getLaplacianface, hw4_q0_faces

%   Homework 4 - shared experiment definition.

here = fileparts(mfilename("fullpath"));

cfg = struct();
cfg.dataFile = fullfile(here, "small-face-dataset", "faces-ids-n6680-m417-20x20.mat");
cfg.selfDir = fullfile(here, "self");
cfg.celebDir = fullfile(here, "celebrity");
cfg.facesFile = fullfile(here, "hw4_faces.mat");
cfg.selfId = 14387365;
cfg.celebId = 418;
cfg.nSelfDb = 10;
cfg.d0 = 32;
cfg.d0Sweep = [8 16 32 64 128];
cfg.nRecog = 800;
cfg.nRetrieve = 400;
cfg.topK = 5;
cfg.nLocal = 4000;
cfg.nQuery = 20;
cfg.kdLocal = 16;
cfg.knnList = [100 200 400];
cfg.seed = 0;
cfg.resultsDir = fullfile(here, "results");
end
