function [x0, y_max] = getScaleResponseExtrema(x, y)
%getScaleResponseExtrema - Peak of a LoG response across scale by cubic fit
%   [x0, y_max] = getScaleResponseExtrema(x, y) fits the cubic
%   f(x) = a*x^3 + b*x^2 + c*x + d to the responses y measured at the
%   scales x, and returns the scale x0 at which the fitted response peaks
%   together with the fitted peak value y_max = f(x0).
%
%   The peak is found from the gradient. f'(x) = 3a*x^2 + 2b*x + c is a
%   quadratic, so it has at most two roots. A root is accepted as the
%   extremum when it is real, lies strictly inside [min(x), max(x)], has
%   f''(x) < 0, which makes it a maximum and not a minimum, and is at
%   least as high as the fitted curve at both ends of the range.
%
%   If no root passes, the response has no peak inside the sampled range.
%   Its largest value is at the smallest or the largest scale. x0 and
%   y_max are then both NaN, which tells the caller that this is not a
%   scale space extremum.
%
%   x and y must be vectors of the same length with at least 4 elements.
%
%   Example: Response that peaks between two sampled scales
%       x = [1.200 1.518 1.920 2.429 3.072 3.886 4.915];
%       y = [16 21 24 22 16 9 6];
%       [x0, y_max] = getScaleResponseExtrema(x, y)
%
%   See also polyfit, polyder, roots, polyval, getLoGKeyPoints

%   Homework 2, Q2 - scale response extrema.

arguments
    x (1,:) double
    y (1,:) double
end

if numel(x) ~= numel(y)
    error("getScaleResponseExtrema:sizeMismatch", ...
        "x and y must have the same number of elements.");
end
if numel(x) < 4
    error("getScaleResponseExtrema:tooFewPoints", ...
        "A cubic needs at least 4 points, %d given.", numel(x));
end

x0 = NaN;
y_max = NaN;

% y = a*x^3 + b*x^2 + c*x + d
p = polyfit(x, y, 3);
dp = polyder(p);
ddp = polyder(dp);

% Critical points are where the gradient is zero.
r = roots(dp);
r = r(abs(imag(r)) < 1e-12 * max(1, abs(real(r))));
r = real(r);

isInside = (r > min(x)) & (r < max(x));
isMaximum = polyval(ddp, r) < 0;
r = r(isInside & isMaximum);

% A cubic can turn over once inside the range and still be higher at one
% of the ends. That bump is not the peak of the response, so reject it.
endValues = polyval(p, [min(x) max(x)]);
r = r(polyval(p, r) >= max(endValues));

% A cubic has at most one local maximum, so r is empty or a scalar.
if ~isempty(r)
    x0 = r(1);
    y_max = polyval(p, x0);
end
end
