function [A0, eigv] = getEigenface(training_data, kd)
%getEigenface - PCA face model, the Eigenface basis
%   A0 = getEigenface(training_data, kd) returns the first kd principal
%   directions of the training faces as the columns of the 400-by-kd
%   matrix A0, ordered by decreasing variance. training_data is either
%   an n-by-400 matrix of faces, one 20-by-20 image per row in
%   column-major order, or a structure with that matrix in the field
%   faces. A face f, a 1-by-400 row, is represented by f*A0.
%
%   The projection is f*A0 without subtracting the mean face, as in the
%   course code. Subtracting it would shift every face by the same
%   amount, so distances between faces and the recognition results do
%   not change.
%
%   [A0, eigv] = getEigenface(...) also returns all the eigenvalues of
%   the covariance of the faces, largest first.
%
%   Example: The first eight Eigenfaces of the data set
%       load small-face-dataset/faces-ids-n6680-m417-20x20.mat
%       A0 = getEigenface(faces, 8);
%       montage(reshape(rescale(A0), 20, 20, 1, 8))
%
%   See also pca, getFisherface, getLaplacianface

%   Homework 4, Q1 - Eigenface.

arguments
    training_data
    kd (1,1) double {mustBeInteger, mustBePositive}
end

faces = iFaces(training_data);
if kd > size(faces, 2)
    error("getEigenface:kdTooLarge", ...
        "kd = %d is more than the %d pixels of a face.", kd, size(faces, 2));
end

% princomp was removed from MATLAB. pca returns the same directions.
[coeff, ~, eigv] = pca(faces);
A0 = coeff(:, 1:kd);
end

function faces = iFaces(training_data)
%iFaces - The face matrix from a matrix or a structure
if isstruct(training_data)
    faces = double(training_data.faces);
else
    faces = double(training_data);
end
end
