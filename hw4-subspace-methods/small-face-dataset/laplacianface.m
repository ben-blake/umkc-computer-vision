%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EE 401/5590 Special Topics: Image Analysis & Retrieval
%  LPP and Laplacian face 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%clear;
% download from: https://umkc.box.com/s/2mwj2nrlp8ftg0omq695qt943cx3esk4
load ../../Grassmann/matlab/data/faces-ids-n6680-m417-20x20.mat;
path(path, '../../tools/LPP');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% do PCA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[A1, s, lat]=princomp(faces); 

h=20; w=20;

figure(30); 
subplot(1,2,1); grid on; hold on; stem(lat, '.'); 
f_eng=lat.*lat; 
subplot(1,2,2); grid on; hold on; plot(cumsum(f_eng)/sum(f_eng), '.-'); 


figure(31); 
for k=1:8
    subplot(2,4,k); colormap('gray'); imagesc(reshape(A1(:,k), [h, w]));
    title(sprintf('eigf_%d', k));
end

%LPP
n_face = 1200; n_subj = length(unique(ids(1:n_face))); 

% eigenface 
kd=32; x1 = faces(1:n_face,:)*A1(:,1:kd); ids=ids(1:n_face); 

% LPP - compute affinity
f_dist1 = pdist2(x1, x1);
% heat kernel size
mdist = mean(f_dist1(:)); h = -log(0.15)/mdist; 
S1 = exp(-h*f_dist1); 
figure(32); subplot(2,2,1); imagesc(f_dist1); colormap('gray'); title('d(x_i, d_j)');
subplot(2,2,2); imagesc(S1); colormap('gray'); title('affinity'); 
%subplot(2,2,3); grid on; hold on; [h_aff, v_aff]=hist(S(:), 40); plot(v_aff, h_aff, '.-'); 
% utilize supervised info
id_dist = pdist2(ids, ids);
subplot(2,2,3); imagesc(id_dist); title('label distance');
S2=S1; S2(find(id_dist~=0)) = 0; 
subplot(2,2,4); imagesc(S1); colormap('gray'); title('affinity-supervised');  



% laplacian face
lpp_opt.PCARatio = 1; 
[A2, eigv2]=LPP(S2, lpp_opt, x1); 


eigface = eye(400)*A1(:,1:kd);
lapface = eye(400)*A1(:,1:kd)*A2; 
for k=1:8
   figure(36);
   subplot(2,4,k); imagesc(reshape(eigface(:,k),[20, 20])); colormap('gray');
   title(sprintf('eigf_%d', k)); 
   figure(37);
   subplot(2,4,k); imagesc(reshape(lapface(:,k),[20, 20])); colormap('gray');
   title(sprintf('lapf_%d', k)); 
end

x2 = x1*A2; 
f_dist2 = pdist2(x2, x2);

figure(38); grid on; hold on;
% for subj=1
d0 = f_dist1(1:7,1:7); d1=f_dist1(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-k', 'DisplayName', 'eigenface kd=32');

d0 = f_dist2(1:7,1:7); d1=f_dist2(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-r', 'DisplayName', 'fisher kd=32');

xlabel('fpr'); ylabel('tpr'); title(sprintf('eigen vs fisher face recog: %d people, %d faces',n_subj, n_face));
legend('eigen kd=32', 'laplacian kd=32', 0); axis([0 1 0 1]);



% mAP performance


return;
