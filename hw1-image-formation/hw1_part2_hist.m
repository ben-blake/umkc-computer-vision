%hw1_part2_hist - Compute the HSV histogram features of the data set
%   For each code book size K and each distance metric, quantizes all
%   600 images against the code book and stores the normalized
%   histograms in an N-by-K matrix. The results are saved as
%   hsv_hist64.mat, hsv_hist128.mat (Euclidean) and hsv_hist64_kld.mat,
%   hsv_hist128_kld.mat (KL divergence), each holding hsv_hist and
%   labels. Already existing MAT files are skipped, so the script can be
%   run one code book at a time.
%
%   See also getHSVHist, hw1_part1_colortable, hw1_part3_recog

%   Homework 1, Part 2.

if ~exist("KList", "var")
    KList = [64 128];               % override before running to split
end
[files, labels, classes] = hw1_config();
N = numel(files);

for K = KList
    load(sprintf("color_table%d.mat", K), "color_table");
    for opt = ["euc" "kld"]
        suffix = "";
        if opt == "kld"
            suffix = "_kld";        % Euclidean files keep the plain name
        end
        matFile = sprintf("hsv_hist%d%s.mat", K, suffix);
        if isfile(matFile)
            fprintf("%s already exists, skipping\n", matFile);
            continue
        end
        fprintf("K = %3d, opt = %s ... ", K, opt);
        t0 = tic;
        hsv_hist = zeros(N, K);
        for i = 1:N
            hsv_hist(i, :) = getHSVHist(imread(files(i)), color_table, opt);
        end
        fprintf("%.1f s -> %s\n", toc(t0), matFile);
        save(matFile, "hsv_hist", "labels", "classes", "color_table");
    end
end
