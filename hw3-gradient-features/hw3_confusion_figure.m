function fig = hw3_confusion_figure(cms, accs, cfg, t, agg)
%hw3_confusion_figure - Six confusion maps of one feature in one figure
%   fig = hw3_confusion_figure(cms, accs, cfg, t, agg) draws the confusion
%   maps of feature type cfg.types(t) for all kd and nc, as a grid with
%   one row per kd and one column per nc. cms is a cell array and accs a
%   numeric array, both indexed (t, kd index, nc index). agg is the name
%   of the aggregation used in the titles, "fv" or "vlad".
%
%   Each map is row normalized and drawn in gray, white for 100%, with
%   the true class down the rows and the predicted class across, in the
%   style of the example in the assignment.
%
%   See also hw3_loo, hw3_q3_fisher, hw3_q4_vlad, confusionmat

%   Homework 3 - confusion map figure.

nKd = numel(cfg.kdList);
nNc = numel(cfg.ncList);
names = strrep(cfg.classes, "_", " ");

fig = figure(Name=cfg.typeNames(t) + " " + agg, Position=[60 60 1500 900]);
tl = tiledlayout(fig, nKd, nNc, TileSpacing="compact", Padding="compact");
for a = 1:nKd
    for b = 1:nNc
        cm = cms{t, a, b};
        ax = nexttile(tl);
        imagesc(ax, cm ./ sum(cm, 2), [0 1]);
        colormap(ax, "gray"); axis(ax, "image");
        yticks(ax, 1:numel(names)); xticks(ax, 1:numel(names));
        xticklabels(ax, []);
        if b == 1
            yticklabels(ax, names);
        else
            yticklabels(ax, []);
        end
        if a == nKd
            xticklabels(ax, names); xtickangle(ax, 60);
        end
        ax.FontSize = 8;
        ax.Toolbar.Visible = "off";     % keeps it out of exportgraphics
        title(ax, sprintf("%s %s(%d, %d) accuracy: %.3f", cfg.types(t), agg, ...
            cfg.kdList(a), cfg.ncList(b), accs(t, a, b)), FontSize=10);
    end
end
cb = colorbar(ax);
cb.Layout.Tile = "east";
cb.Label.String = "share of the true class";
title(tl, sprintf("%s %s: leave-one-out 1-NN confusion maps, rows true class, columns predicted", ...
    cfg.typeNames(t), upper(agg)));
theme(fig, "light");
end
