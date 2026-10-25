%hw3_q2_gmm - PCA and GMM models of the three feature spaces
%   Samples up to cfg.perImage features from every image, computes the PCA
%   of each feature space, plots its eigenvalues, and fits a diagonal GMM
%   with getFisherVectorModel for every combination of kd = [16 32] and
%   nc = [32 64 128]. The 18 models are saved to gmm_models.mat.
%
%   See also getFisherVectorModel, hw3_q1_features, hw3_q3_fisher

%   Homework 3, Q2.

clear; clc;
cfg = hw3_config();
load gradient_features.mat f_sift f_dsift f_hog
allFeats = {f_sift, f_dsift, f_hog};
N = numel(cfg.files);
nType = numel(cfg.types);
nKd = numel(cfg.kdList);
nNc = numel(cfg.ncList);

% The same number of features from every image, so that images with many
% SIFT key points do not dominate the model. The GMM never sees a label.
rng(cfg.seed);
samples = cell(1, nType);
for t = 1:nType
    pooled = cell(N, 1);
    for i = 1:N
        fi = allFeats{t}{i};
        k = min(cfg.perImage, size(fi, 1));
        pooled{i} = double(fi(randperm(size(fi, 1), k), :));
    end
    samples{t} = vertcat(pooled{:});
    fprintf("%-11s sample %d by %d\n", cfg.typeNames(t), size(samples{t}));
end

matFile = "gmm_models.mat";
if isfile(matFile)
    fprintf("Loading models from %s\n", matFile);
    load(matFile, "models", "A", "lat", "fitSeconds");
else
    A = cell(1, nType);
    lat = cell(1, nType);
    models = cell(nType, nKd, nNc);
    fitSeconds = zeros(nType, nKd, nNc);
    for t = 1:nType
        for a = 1:nKd
            for b = 1:nNc
                kd = cfg.kdList(a); nc = cfg.ncList(b);
                tStart = tic;
                [gmm, At, latT] = getFisherVectorModel(samples{t}, kd, nc, ...
                    Seed=cfg.seed);
                fitSeconds(t, a, b) = toc(tStart);
                models{t, a, b} = gmm;
                fprintf("%-11s kd = %2d, nc = %3d: %5.1f s, %3d iterations\n", ...
                    cfg.typeNames(t), kd, nc, fitSeconds(t, a, b), gmm.iterations);
                if ~gmm.converged
                    fprintf("    did not converge in the iteration limit\n");
                end
            end
        end
        A{t} = At;          % pca does not depend on kd or nc
        lat{t} = latT;
    end
    save(matFile, "models", "A", "lat", "fitSeconds");
end
iterations = cellfun(@(g) g.iterations, models);
converged = cellfun(@(g) g.converged, models);
fprintf("\n%d of %d models converged, %d to %d EM iterations\n", ...
    sum(converged(:)), numel(converged), min(iterations(:)), max(iterations(:)));

% Variance kept by the first kd directions.
fprintf("\n%-11s %4s %10s %10s %10s %10s\n", "feature", "d", ...
    "kd = 16", "kd = 32", "95% at", "99% at");
explained = zeros(nType, nKd);
for t = 1:nType
    c = cumsum(lat{t}) / sum(lat{t});
    explained(t, :) = c(cfg.kdList);
    fprintf("%-11s %4d %9.1f%% %9.1f%% %10d %10d\n", cfg.typeNames(t), ...
        numel(lat{t}), 100*explained(t, :), find(c >= 0.95, 1), find(c >= 0.99, 1));
end

fig = figure(Name="PCA eigenvalues", Position=[100 100 1100 420]);
tl = tiledlayout(fig, 1, 2, TileSpacing="compact", Padding="compact");
ax = nexttile(tl);
for t = 1:nType
    semilogy(ax, lat{t} / sum(lat{t}), "-", LineWidth=1.4); hold(ax, "on");
end
xline(ax, cfg.kdList, ":", "kd = " + string(cfg.kdList), LabelVerticalAlignment="bottom");
grid(ax, "on"); xlim(ax, [1 128]);
xlabel(ax, "principal component"); ylabel(ax, "eigenvalue / sum of eigenvalues");
legend(ax, cfg.typeNames, Location="northeast");
title(ax, "eigenvalues, each as a share of the total variance");
ax = nexttile(tl);
for t = 1:nType
    plot(ax, 100 * cumsum(lat{t}) / sum(lat{t}), "-", LineWidth=1.4); hold(ax, "on");
end
xline(ax, cfg.kdList, ":", "kd = " + string(cfg.kdList), LabelVerticalAlignment="bottom");
grid(ax, "on"); xlim(ax, [1 128]); ylim(ax, [0 100]);
xlabel(ax, "number of components kept"); ylabel(ax, "% of variance");
legend(ax, cfg.typeNames, Location="southeast");
title(ax, "cumulative variance");
title(tl, "Q2: PCA eigenvalues of the three feature spaces");
theme(fig, "light");
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, "q2_eigenvalues.png"), Resolution=150);

% How spread out the components of each model are: the share of the
% largest weight, and how many components carry 90% of the weight.
fprintf("\n%-11s %4s %4s %10s %10s %10s\n", "feature", "kd", "nc", ...
    "max prior", "90% in", "-logL/n");
gmmStats = zeros(nType, nKd, nNc, 3);
for t = 1:nType
    for a = 1:nKd
        for b = 1:nNc
            g = models{t, a, b};
            p = sort(g.prior, "descend");
            gmmStats(t, a, b, :) = [p(1), find(cumsum(p) >= 0.9, 1), ...
                g.negLogLikelihood / size(samples{t}, 1)];
            fprintf("%-11s %4d %4d %10.3f %10d %10.2f\n", cfg.typeNames(t), ...
                cfg.kdList(a), cfg.ncList(b), squeeze(gmmStats(t, a, b, :)));
        end
    end
end

save hw3_q2_results.mat explained gmmStats fitSeconds iterations converged
