function hw4_save_figure(fig, name)
%hw4_save_figure - Export a figure to results/ the way every script does
%   hw4_save_figure(fig, name) gives the figure fig the light theme,
%   hides the axes toolbars so they do not end up in the image, and
%   writes it to results/name at 150 dpi.
%
%   Example: Save a plot
%       fig = figure; plot(1:10);
%       hw4_save_figure(fig, "example.png")
%
%   See also exportgraphics, hw4_config

%   Homework 4 - figure export.

arguments
    fig (1,1) matlab.ui.Figure
    name (1,1) string
end

cfg = hw4_config();
theme(fig, "light");
for ax = findall(fig, "Type", "axes")'
    ax.Toolbar.Visible = "off";
end
drawnow;
exportgraphics(fig, fullfile(cfg.resultsDir, name), Resolution=150);
end
