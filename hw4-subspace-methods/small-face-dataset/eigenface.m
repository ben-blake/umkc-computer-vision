%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EE 401/5590 Special Topics: Image Analysis & Retrieval
%  Eigenface & Fisherface 
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


kd=16; nface=160;

x=faces*A1(:, 1:kd); 
f_dist = pdist2(x(1:nface,:), x(1:nface,:));
figure(32); 
imagesc(f_dist); colormap('gray');

figure(33); hold  on; grid on; 

d0 = f_dist(1:7,1:7); d1=f_dist(8:end, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-r', 'DisplayName', 'tpr-fpr color, data set 1');
xlabel('fpr'); ylabel('tpr'); title('eig face recog');
legend('kd=8', 'kd=12', 'kd=16', 0);


figure(34); hold on; grid on; 
styl = ['*r'; 'ob'; '+k'; '^m'];
for k=1:4
    figure(34); offs = find(ids==k); plot3(x(offs, 1), x(offs,2), x(offs, 3), styl(k,:) );
    figure(35); subplot(2,2,k); imagesc(reshape(faces(offs(1),:), [h, w])); colormap('gray');
end


% Fisherface
% path(path, '../../tools/LPP');
n_face = 600; n_subj = length(unique(ids(1:n_face))); 
%eigenface kd
kd = 32;
opt.Fisherface = 1; 
[A2, lat]=LDA(ids(1:n_face), opt, faces(1:n_face,:)*A1(:,1:kd));
% eigenface
x1 = faces*A1(:,1:kd); 
f_dist1 = pdist2(x1, x1);
% fisherface
x2 = faces*A1(:,1:kd)*A2; 
f_dist2 = pdist2(x2, x2);

figure(36); grid on; hold on;
% for subj=1
d0 = f_dist1(1:7,1:7); d1=f_dist1(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-k', 'DisplayName', 'eigenface kd=32');

d0 = f_dist2(1:7,1:7); d1=f_dist2(1:7, 8:end);
[tp, fp, tn, fn]= getPrecisionRecall(d0(:), d1(:), 40); 
plot(fp./(tn+fp), tp./(tp+fn), '.-r', 'DisplayName', 'fisher kd=32');

xlabel('fpr'); ylabel('tpr'); title(sprintf('eigen vs fisher face recog: %d people, %d faces',n_subj, n_face));

legend('eigen kd=32', 'fisher kd=32', 0);



% mAP performance


return;
