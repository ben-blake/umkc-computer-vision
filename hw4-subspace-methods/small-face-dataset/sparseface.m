%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sparse Representation recovery with L1Magic for face recognition 
% z. li
% 2009.03.10
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;
path(path, 'l1magic/Data');
path(path, 'l1magic/Optimization');
facepath='data/att_faces/';
%%%%%%%%%%%%%%%%%%
% sparse demo from L1 magic
%%%%%%%%%%%%%%%%%%
if (1)
figure(1);
% load random states for repeatable experiments
load RandomStates
rand('state', rand_state);
randn('state', randn_state);
% signal length
N = 512;
% number of spikes in the signal
T = 20;
% number of observations to make
K = 120;
% random +/- 1 signal
x = zeros(N,1);
q = randperm(N);
x(q(1:T)) = sign(randn(T,1));
subplot(3,1,1); plot(x); title('x(t)'); axis([1 500 -1.2 1.2]);
% measurement matrix
fprintf('\n Creating measurment matrix...');
A = randn(K,N);
% othorgonalize 
A = orth(A')';

% observations
y = A*x;
% initial guess = min energy
x0 = A'*y;
subplot(3,1,2); plot(x0); title('x_0(t)'); axis([1 500 -1.2 1.2]);
% solve with primal-dual method
xp = l1eq_pd(x0, A, [], y, 1e-3);

subplot(3,1,3); plot(xp); title('x(t) recovered by L1 magic'); axis([1 500 -1.2 1.2]);

% test l1magic
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% random faces
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (1)

plotface = 0;
loadFaceIcons = 1;
loadNonFaceIcons = 1;
%load AT&T faces
nSubj=40; nPic=10;
% icon size
h=12; w=10;

if loadFaceIcons
    %load face-icons-36x30.mat;
    %load face-icons-60x50.mat;
    %load face-icons-24x20.mat;
    load face-icons-12x10.mat;
else
    % face icons
    icons = cell(nSubj, nPic); 
    %count 
    t=1;
    figure(2); colormap('gray');
    for j=1:nSubj
        for k=1:nPic
            fim_name=sprintf('%ss%d/%d.pgm', facepath, j, k);
            im = imread(fim_name);
            icn = imresize(im, [h, w], 'bilinear');
            icn = double(icn);
            icons{j, k} = icn;
            faces(t,1:w*h) = (icn(:))'; t=t+1;
            if plotface
                subplot(2, 5, k); imagesc(icons{j,k}); axis off;
            end
        end
    end
    % save data
    % save face-icons-36x30.mat icons faces -mat;
    % save face-icons-60x50.mat icons faces -mat;
    % save face-icons-24x20.mat icons faces -mat;
    save face-icons-12x10.mat icons faces -mat;
end

% add more icons from non-face images
if (loadNonFaceIcons)
        load nfaces-12x10-1200.mat; 
else
    im1 = imread('D:\zli\pic\2007\allerton07-pic\cimg5475.jpg');
    im2 = rgb2gray(im1);
    [im_h, im_w]=size(im2); 
    nx = 40; ny = 30; t=1;
    for j=1:nx
        for k=1:ny
            xpos = fix(1 + (im_w-w).*rand(1)); 
            ypos = fix(1 + (im_h-h).*rand(1)); 
            nf_icn = double(im2(ypos:ypos+h-1, xpos:xpos+w-1));
            nfaces(t, :) = nf_icn(:)';
            t=t+1;
        end
    end
    % save nfaces-12x10-1200.mat nfaces;
end

% create our measure matrix A: face + nonface icons: dictionary
A=zeros(1600, w*h); A(1:400, :) = faces; A(401:1600, :) = nfaces;
[N, dim]=size(A);
% in col vec form
A = A';

% pick a face: offs in 1-400
figure(3); colormap('gray');
% query face: y
offs = 10; y = faces(offs, :)'; 
subplot(2,2,1); axis off; imagesc(reshape(y, h,w)); title('\fontsize{11}original');

% solve for xp = min |x|_1, s.t. y=Ax
% initial guess = min energy
x0 = A'*y;
% solve with primal-dual method
xp = l1eq_pd(x0, A, [], y, 1e-3);

% normalize
x0 = x0./norm(x0);
xp = xp./norm(xp);

% reconstructed face
yp = A*xp;
subplot(2,2,2); axis off; imagesc(reshape(yp, h,w)); title('\fontsize{11}sparse reconstruction');

% sparse coefficients
subplot(2,2,3); plot(x0); title('\fontsize{11} L_2 recovery of x'); grid on; axis([1 N 0 0.15]);
subplot(2,2,4); plot(xp); title('\fontsize{11} L_1 recovery of x'); grid on; axis([1 N 0 1]);


figure;
im = imread('cameraman.tif');
bw = edge(im, 'log');
bw2 = bwmorph(bw, 'skel', inf); 
subplot(2,2,1); imshow(im);
subplot(2,2,2); imshow(bw);
subplot(2,2,3); imshow(bw2);


end;


