%hw1_part3_recog - Image recognition experiment on the HSV histograms
%   Classifies the 15-class NWPU subset with k-NN on the HSV histogram
%   features from Part 2, under two protocols:
%
%     split - the first 10 images of every class are the test set and
%             the remaining 30 are the training set, as in the
%             assignment skeleton, giving 150 test and 450 training
%             images
%     loo   - leave one out over all 600 images
%
%   Reports overall accuracy for every feature set, per-class accuracy
%   for the best one, and saves a confusion chart.
%
%   See also knnclassify, getHSVHist, hw1_part2_hist
 
%   Homework 1, Part 3.
 
clear; clc;
featSets = ["hsv_hist64" "hsv_hist64_kld" "hsv_hist128" "hsv_hist128_kld"];
kNNList = [1 3 5];
n_class = 15;
n_img = 40;
test_set = 1:10;
train_set = 11:40;
 
% Index bookkeeping. The training set is every image that is not in the
% test set of its own class, so the two sets never overlap.
% Transposed before reshaping so the indices run class by class, which
% is the order the per-class loop below fills its predictions in.
test_idx = reshape(((0:n_class-1)'*n_img + test_set)', [], 1);
train_idx = setdiff(1:n_class*n_img, test_idx)';
 
accSplit = zeros(numel(featSets), numel(kNNList));
accLoo = zeros(numel(featSets), 1);
perClass = zeros(numel(featSets), n_class);
recAll = cell(numel(featSets), 1);
 
for f = 1:numel(featSets)
    load(featSets(f) + ".mat", "hsv_hist", "labels", "classes");
    fprintf("%-18s %d x %d\n", featSets(f), size(hsv_hist, 1), ...
        size(hsv_hist, 2));
 
    % Protocol 1: fixed 10 / 30 split, per class as in the skeleton.
    for kk = 1:numel(kNNList)
        rec = zeros(size(test_idx));
        for k = 1:n_class
            offs = (k-1)*n_img;
            % test: offs + test_set, training: all samples minus test set
            rec_label = knnclassify(hsv_hist(offs+test_set, :), ...
                hsv_hist(train_idx, :), labels(train_idx), kNNList(kk));
            rec((k-1)*numel(test_set) + (1:numel(test_set))) = rec_label;
            if kNNList(kk) == 1
                perClass(f, k) = mean(rec_label == k);
            end
        end
        accSplit(f, kk) = mean(rec == labels(test_idx));
        if kNNList(kk) == 1
            recAll{f} = rec;
        end
    end
 
    % Protocol 2: leave one out over the whole data set.
    N = size(hsv_hist, 1);
    recLoo = zeros(N, 1);
    for i = 1:N
        tr = [1:i-1, i+1:N];
        recLoo(i) = knnclassify(hsv_hist(i, :), hsv_hist(tr, :), labels(tr));
    end
    accLoo(f) = mean(recLoo == labels);
end
 
% Accuracy summary. Chance level is 1/15 = 6.7 percent.
fprintf("\n=== Overall accuracy, 10/30 split (150 test images) ===\n");
disp(array2table(accSplit, RowNames=featSets, ...
    VariableNames="k=" + string(kNNList)));
fprintf("=== Overall accuracy, leave one out (600 images) ===\n");
disp(array2table(accLoo, RowNames=featSets, VariableNames="1-NN"));
 
[bestAcc, best] = max(accSplit(:, 1));
fprintf("Best 1-NN split accuracy: %.1f%% with %s\n", ...
    100*bestAcc, featSets(best));
fprintf("\n=== Per-class accuracy, 1-NN, 10/30 split ===\n");
disp(array2table(100*perClass', RowNames=classes, VariableNames=featSets));
 
% Extra experiment: change the distance the classifier uses, but leave
% the features alone. Euclidean weights a large bin and a small bin the
% same, while cityblock and chi-square are the usual choices for
% comparing histograms.
metrics = ["euclidean" "cityblock" "chisq"];
accMetric = zeros(numel(featSets), numel(metrics));
for f = 1:numel(featSets)
    load(featSets(f) + ".mat", "hsv_hist", "labels");
    for m = 1:numel(metrics)
        rec = knnclassify(hsv_hist(test_idx, :), hsv_hist(train_idx, :), ...
            labels(train_idx), 1, metrics(m));
        accMetric(f, m) = mean(rec == labels(test_idx));
    end
end
fprintf("\n=== Overall accuracy by k-NN metric, 1-NN, 10/30 split ===\n");
disp(array2table(100*accMetric, RowNames=featSets, VariableNames=metrics));
 
% Underscores would render as TeX subscripts in the chart labels.
showClasses = strrep(classes, "_", " ");
fig = figure(Name="Confusion matrix", Position=[100 100 760 640]);
cm = confusionchart(fig, showClasses(labels(test_idx)), ...
    showClasses(recAll{best}));
cm.Title = sprintf("Part 3: %s, 1-NN, %.1f%% accuracy", ...
    strrep(featSets(best), "_", " "), 100*bestAcc);
cm.RowSummary = "row-normalized";
sortClasses(cm, showClasses);
theme(fig, "light");
drawnow;
exportgraphics(fig, "results/part3_confusion.png", Resolution=150);
 
save("hw1_part3_results.mat", "accSplit", "accLoo", "accMetric", ...
    "perClass", "featSets", "kNNList", "metrics", "classes");
