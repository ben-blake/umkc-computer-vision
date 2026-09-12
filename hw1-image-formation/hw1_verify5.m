%hw1_verify5 - Independent checks on the Part 2 homography code
%   Tests getHomographySVD against a case whose answer is known in
%   advance, confirms the identities the write-up claims, and checks that
%   the transpose handed to maketform really is the one imtransform
%   wants. Prints PASS or FAIL for each check.
%
%   See also getHomographySVD, hw1_part2_homography, maketform

%   Homework 1, Part 2 - verification harness.

clear; clc; rng(0);
chk = @(tf, name) fprintf("%-62s %s\n", name, ...
    string(tf).replace("true","PASS").replace("false","FAIL"));

fprintf("--- recovering a homography that is known in advance ---\n");
Htrue = [0.9 -0.3 40; 0.25 1.1 -15; 2e-4 -1e-4 1];
p = [rand(8,1)*600, rand(8,1)*400];
q = (Htrue * [p ones(8,1)]')';
q = q(:,1:2) ./ q(:,3);
[H, sv, h, A] = getHomographySVD(p, q);
Hn = Htrue / Htrue(3,3);
chk(max(abs(H(:) - Hn(:))) < 1e-9, "exact data recovers the true H to 1e-9");
chk(isequal(size(A), [16 9]), "A is 16 by 9 for M = 8");
chk(sv(9) < 1e-9 * sv(1), "ninth singular value is numerically zero");
chk(rank(A) == 8, "A has rank 8 at MATLAB's default tolerance");
chk(abs(norm(h) - 1) < 1e-12, "null vector has unit norm");
% On exact data s9 sits at round-off, where eps*norm(A) is about 1e-10,
% so the two agree relatively rather than absolutely. The real cases at
% the bottom of this file test the identity where s9 is well above noise.
chk(abs(norm(A*h) - sv(9)) <= 1e-9*sv(1), "norm(A*h) equals s9 to round-off");

fprintf("\n--- the two rows of A encode the right constraint ---\n");
i = 3;
x = p(i,1); y = p(i,2); xp = q(i,1); yp = q(i,2);
chk(isequal(A(2*i-1,:), [x y 1 0 0 0 -xp*x -xp*y -xp]), "odd row matches the x equation");
chk(isequal(A(2*i,:),   [0 0 0 x y 1 -yp*x -yp*y -yp]), "even row matches the y equation");
hTrue = reshape(Hn', [9 1]);
chk(norm(A*hTrue) / norm(hTrue) < 1e-9, "the true h lies in the null space of A");

fprintf("\n--- reshape order and scaling ---\n");
chk(isequal(reshape(h, [3 3])' * sign(h(1)), H * norm(h) * sign(h(1)) / norm(h) * H(3,3) / H(3,3)) ...
    || max(abs(reshape(h,[3 3])'/h(9) - H), [], "all") < 1e-9, ...
    "H is reshape(h,[3 3])' divided by h33");

fprintf("\n--- noise makes s9 grow but keeps the fit usable ---\n");
qn = q + 0.3*randn(size(q));
[Hnoise, svn] = getHomographySVD(p, qn);
w = (Hnoise * [p ones(8,1)]')'; w = w(:,1:2)./w(:,3);
chk(svn(9) > sv(9), "noise lifts s9 off zero");
chk(max(vecnorm(w - qn, 2, 2)) < 2, "noisy fit still lands within 2 px");

fprintf("\n--- maketform gets the transpose it expects ---\n");
T = maketform("projective", H');
pts = [10 20; 300 150; 500 400];
viaTform = tformfwd(T, pts);
z = (H * [pts ones(3,1)]')'; viaH = z(:,1:2)./z(:,3);
chk(max(abs(viaTform - viaH), [], "all") < 1e-9, ...
    "tformfwd(maketform('projective',H')) equals H*p");

fprintf("\n--- normalized DLT agrees on clean data ---\n");
Hnorm = getHomographySVD(p, q, Normalize=true);
chk(max(abs(Hnorm - Hn), [], "all") < 1e-8, "Hartley normalization gives the same answer");

fprintf("\n--- the reported results reproduce from the saved matrices ---\n");
load hw1_part2_results.mat res
for i = 1:numel(res)
    r = res(i);
    [H2, sv2] = getHomographySVD(r.b1, r.b2);
    ok = max(abs(H2 - r.H), [], "all") < 1e-9 && max(abs(sv2 - r.sv)) < 1e-9;
    chk(ok, sprintf("pair %d %s reproduces from its 8 stored points", r.pair, r.which));
    chk(abs(norm(r.A*r.h) - r.sv(9)) < 1e-9, ...
        sprintf("pair %d %s residual equals s9", r.pair, r.which));
end
