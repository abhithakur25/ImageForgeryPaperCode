clear all;
close all;
clc


%% Input Image from Database

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
original123=imread([pathname filename]);
original=original123;
figure,imshow(original);
title('input image');


%% color convertion

f=rgb2ycbcr(original);
figure,imshow(f);
title('color converted imag');
load my_regioncoordinates

% nBins=5;
% winSize=7;
% nClass=6;
colormap('default');
a = original;
a = im2bw(a,0.5);
siz = size(a);
%% figure, imshow(a)
    a = bwmorph(a,'clean');
%% figure, imshow(a), title('Result of clean Image');
cform = makecform('srgb2lab');
J = applycform(original,cform);
%% figure;imshow(J);
img1 = imfilter(J,ones(3,3)/9);
figure; imshow(img1)
img=rgb2gray(img1);
[row, col]=size(img);
[labels, num] = slicomex(img1,img);%numlabels is the same as number of superpixels
labels=labels+1;
immm=drawregionboundaries(labels,img1,[255 0 0]);
figure; imshow(immm)

%%
tot_fpts=0;
desc=cell(1,num);
loc=cell(1,num);
for i=1:num
    mask=labels~=i;
    imageFile=img;
        imageFile(mask)=0;
%         [locs,descriptors] = siftMatch( imageFile, img1 ) ;
    [locs,descriptors] = mift_sift( imageFile, 'Verbosity', 0 ) ;
    clear imageFile
    desc{1,i}=descriptors';
    loc{1,i}=locs';
end
%% matching between patches
th_dist=0.1;
INDEX_inf=cell(1,num*(num-1)/2);
idx=1;
for i=1:num-1
    descA=desc{1,i};
    if ~isempty(descA)
        inx=1;
        for j=i+1:num
            descB=desc{1,j};
            if ~isempty(descB)
                    INFO_mat=zeros(10*size(descB,1),5);
                    [index, distance] = knnsearch_base(descB, descA, 3) ;   % base-MATLAB k-NN (no Stats Toolbox)
                    k=1:size(distance,1);
                    k=repmat(k',1,size(distance,2));
                    k=reshape(k,numel(k),1);
                    distance=reshape(distance,numel(distance),1);
                    index=reshape(index,numel(index),1);
                    index=index((0<distance)&(distance<th_dist));
                    k=k((0<distance)&(distance<th_dist));
                    distance=distance((0<distance)&(distance<th_dist));
                    if ~isempty(distance)
                    len=size(distance,1)-1;
                    INFO_mat(inx:inx+len,1)=i;                   %patch a
                    INFO_mat(inx:inx+len,2)=j;                   %patch b
                    INFO_mat(inx:inx+len,3)=k;                   %keypoint x of patch a
                    INFO_mat(inx:inx+len,4)=index;               %keypoint y of patch b
                    INFO_mat(inx:inx+len,5)=distance;
                    inx=inx+len+1;
                    end
               INFO_mat( all(~INFO_mat,2), : ) = [];
               INDEX_inf{1,idx}=INFO_mat;
               clear INFO_mat
               clear k
               idx=idx+1;
            end
            clear descB
        end
    end
    clear descA
end
%%
 INDEX_inf=INDEX_inf(~cellfun('isempty',INDEX_inf));
 L=size(INDEX_inf,2);
 matched_pts=[];
 M_loc=cell(1,L);
for i=1:L
    INFO_mat=INDEX_inf{1,i};
    locA=loc{1,INFO_mat(1,1)};
    locB=loc{1,INFO_mat(1,2)};
    matched_loc=[locA(INFO_mat(:,3),[1 2]) locB(INFO_mat(:,4),[1 2])];
    matched_loc=unique(matched_loc,'rows');
    tot_fpts=tot_fpts+size(matched_loc,1);
    for j=1:size(matched_loc,1);
        loc_dist(j)=round(norm(matched_loc(j,[1 2])-matched_loc(j,[3 4]),2));
    end
    M_loc{1,i}=matched_loc;
    if ~isempty(matched_loc)
        ilx=i*ones(size(matched_loc,1),1);
    matched_loc=[matched_loc loc_dist' ilx];
    matched_pts=[matched_pts; matched_loc];
    clear ilx
    end
    clear loc_dist
end
thresh=ceil(tot_fpts/L);
if ~isempty(matched_pts)
prev=0;
while size(matched_pts,1)~=prev
i=1;
wid=1;
prev=size(matched_pts,1);
matched_pts=sortrows(matched_pts,5);
while i<size(matched_pts,1)
    if matched_pts(i+1,5)==matched_pts(i,5)
        wid=wid+1;
    else
        if wid<=thresh
            for tim=1:wid
                j=i-wid+1;
            matched_pts(j,:)=[];
            end
        i=i-wid;
        end
        wid=1;
    end
    i=i+1;
end
if i==size(matched_pts,1) && wid<=thresh
        for tim=1:wid
            j=i-wid+1;
            matched_pts(j,:)=[];
        end
end
wid=1;
i=1;
matched_pts=sortrows(matched_pts,6);
 while i<size(matched_pts,1)
    if matched_pts(i+1,6)==matched_pts(i,6)
        wid=wid+1;
    else
        if wid<=thresh
            for tim=1:wid
                j=i-wid+1;
            matched_pts(j,:)=[];
            end
        i=i-wid;
        end
        wid=1;
    end
    i=i+1;
end
if i==size(matched_pts,1) && wid<=thresh
        for tim=1:wid
            j=i-wid+1;
            matched_pts(j,:)=[];
        end
end
end
end
%%
key_locs=zeros(2*size(matched_pts,1),2);
if ~isempty(matched_pts)
    key_locs(1:size(matched_pts,1),:)=matched_pts(:,[1 2]);
    key_locs(size(matched_pts,1)+1:end,:)=matched_pts(:,[3 4]);
end
%%
keypoints_r=round(key_locs);
baseimgpts=zeros(row,col);
for i=1:size(keypoints_r,1)
    baseimgpts(keypoints_r(i,2),keypoints_r(i,1))=255;
end
figure, imshow(baseimgpts), title('SIFT Features');
ms=baseimgpts;
imgR=double(img1(:,:,1));
imgG=double(img1(:,:,2));
imgB=double(img1(:,:,3));
frgd_rgns=zeros(row,col);

if ~isempty(matched_pts)
matched_pts=round(sortrows(matched_pts,6));
nL=numel(unique(matched_pts(:,6)));
matched_pts(:,[1 2 3 4])=matched_pts(:,[2 1 4 3]);
[IDX,C] = kmeans_base(matched_pts(:,6), nL);   % base-MATLAB k-means (no Stats Toolbox)
for i=1:nL
    G = matched_pts(IDX==i,:);
    labeledImage=zeros(row,col);
    for j=1:size(G,1)
            labeledImage(matched_pts(j,1),matched_pts(j,2))=1;
            labeledImage(matched_pts(j,3),matched_pts(j,4))=1;
    end
    measurementsRed = regionprops(labeledImage, imgR, 'MeanIntensity');
    measurementsGreen = regionprops(labeledImage, imgG, 'MeanIntensity');
    measurementsBlue = regionprops(labeledImage, imgB, 'MeanIntensity');
    TR=measurementsRed.MeanIntensity;
    TG=measurementsGreen.MeanIntensity;
    TB=measurementsBlue.MeanIntensity;
    for j=1:size(G,1)
    frgd_rgns=color_grow(img1,baseimgpts,G(j,[1 2]),G(j,[3 4]),TR,TG,TB);

    end
    clear G
    clear measurementsRed
    clear measurementsGreen
    clear measurementsBlue
end
end
figure, imshow(frgd_rgns);
fprintf('\n Saving ...')
    imwrite(frgd_rgns,'KeyPoints.jpg');

%% ------------------ MORPHOLOGICAL POSTPROCESSING ------------------
ms = frgd_rgns;
ms = imfill(ms, 8, 'holes');


dil = strel('disk',38);
dil = imdilate(ms,dil);

figure, imshow(dil); title('Forgery Detected Image');
imwrite(dil,'Forgerysegmentedimage.jpg');
%% ------------------ PERFORMANCE METRICS ------------------
k1 = double(dil) + 80;

bwim11 = adaptivethreshold(k1,11,0.03,0);
k5 = entropy(bwim11);

if (k5 > 0.01)
    msgbox('FORGERY DETECTED','RESULT OF IMAGE FORGERY');
else
    msgbox('FORGERY NOT DETECTED','RESULT OF IMAGE FORGERY');
end

%% ------------------ DEEP-LEARNING ACCURACY EVALUATION ------------------
% NOTE: The original SVM evaluation here used fitcsvm/cvpartition, which require
% the Statistics & Machine Learning Toolbox (not installed on this machine).
% It has been replaced by the optimized deep-learning model produced by
% optimize_dl2.m (winning configuration "E8_log_4824L":
%   log1p features -> z-score -> dense net [48 24], dropout 0.35,
%   weight-decay 5e-4, class-weighted loss, 800 epochs, 7-net ensemble).
% The model + preprocessing are saved in forgery_dl_model.mat by
% train_final_model.m, and applied through dl_forgery_predict.m.
fprintf('\nEvaluating with optimized deep-learning model (E8_log_4824L)...\n');

% Ensure a trained model exists; train it on first use.
if exist('forgery_dl_model.mat','file') ~= 2
    fprintf('forgery_dl_model.mat not found - training it now (one-time)...\n');
    train_final_model;
end
Mdl = load('forgery_dl_model.mat');

% ---- classify the feature database with the wired-in model ----
Sdb = load('Accuracy_Data.mat','Train_Feat','Train_Label');
Xdb = double(Sdb.Train_Feat);
ydb = double(Sdb.Train_Label(:)); ydb(ydb~=0) = 1;
[predLab, predProb] = dl_forgery_predict(Xdb);   %#ok<ASGLU>

% ---- choose scores for reporting: honest cross-validated if available ----
if ~isempty(Mdl.cvScore) && ~isempty(Mdl.cvY)
    score  = Mdl.cvScore(:);
    ytrue  = Mdl.cvY(:);
    srcTxt = 'cross-validated';
else
    score  = predProb(:);
    ytrue  = ydb;
    srcTxt = 'resubstitution';
end
yhat = double(score >= Mdl.thr);

% ---- confusion + metrics (positive class = 1 = forged) ----
TP = sum(yhat==1 & ytrue==1); FP = sum(yhat==1 & ytrue==0);
TN = sum(yhat==0 & ytrue==0); FN = sum(yhat==0 & ytrue==1);
precision = TP/(TP+FP+eps);
recall    = TP/(TP+FN+eps);
F1        = 2*precision*recall/(precision+recall+eps);
accuracy  = (TP+TN)/numel(ytrue);

% ---- ROC-AUC via Mann-Whitney U statistic (no toolbox needed) ----
nP = sum(ytrue==1); nN = sum(ytrue==0);
if nP>0 && nN>0
    [~,ord] = sort(score); ranks = zeros(numel(score),1); ranks(ord) = 1:numel(score);
    [us,~,ic] = unique(score);
    for kk = 1:numel(us), msk = ic==kk; ranks(msk) = mean(ranks(msk)); end
    AUC = (sum(ranks(ytrue==1)) - nP*(nP+1)/2)/(nP*nN);
else
    AUC = NaN;
end

fprintf('Performance (%s, positive = forged):\n', srcTxt);
fprintf('Precision: %.2f%%\n', precision*100);
fprintf('Recall:    %.2f%%\n', recall*100);
fprintf('F1 Score:  %.2f%%\n', F1*100);
fprintf('ROC-AUC:   %.4f\n',   AUC);
fprintf('Accuracy:  %.2f%%\n', accuracy*100);

C = [TN FP; FN TP];   % confusion matrix (rows actual 0/1, cols predicted 0/1)

% ---- relate metrics to the CURRENT image's detection result ----
if exist('k5','var')
    if k5 > 0.01
        fprintf('Current image verdict: FORGERY DETECTED (region entropy %.4f)\n', k5);
    else
        fprintf('Current image verdict: NOT FORGED (region entropy %.4f)\n', k5);
    end
end

%% ------------------ Grayscale Bar Graph (600 x 500) ------------------
vals = [precision, recall, F1, AUC, accuracy];
labels_txt = {'Precision','Recall','F1','ROC-AUC','Accuracy'};
hBarFig = figure('Name','DL Metrics','Units','pixels','Position',[100 100 600 500]);
colormap(hBarFig,'gray');
hb = bar(vals,'FaceColor','flat','EdgeColor','k'); ylim([0 1.05]); grid on;
gl = [0.15;0.32;0.50;0.68;0.85]; hb.CData = [gl gl gl];
set(gca,'XTick',1:numel(vals),'XTickLabel',labels_txt,'FontSize',10);
ylabel('Score'); title('Performance Metrics (Deep-Learning Model)');
text(1:numel(vals), vals+0.02, compose('%.3f',vals), ...
     'HorizontalAlignment','center','FontSize',9);

%% ------------------ ROC Curve (600 x 500) ------------------
th = sort(unique([0;score;1]),'descend');
fpr = zeros(numel(th),1); tpr = zeros(numel(th),1);
for i = 1:numel(th)
    pr = score >= th(i);
    tpr(i) = sum(pr & ytrue==1)/max(nP,1);
    fpr(i) = sum(pr & ytrue==0)/max(nN,1);
end
fpr = [0;fpr;1]; tpr = [0;tpr;1]; [fpr,oo] = sort(fpr); tpr = tpr(oo);
figure('Name','ROC Curve','Units','pixels','Position',[720 100 600 500]);
plot(fpr,tpr,'k','LineWidth',2); hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;
title(sprintf('ROC Curve (AUC = %.4f)',AUC));

%% ------------------ Confusion Matrix (600 x 500, grayscale) ------------------
hCMFig = figure('Name','Confusion Matrix','Units','pixels','Position',[100 640 600 500]);
imagesc(C); colormap(hCMFig,'gray'); axis equal tight; colorbar;
title('Confusion Matrix (Actual vs Predicted)');
xlabel('Predicted'); ylabel('Actual');
set(gca,'XTick',1:2,'XTickLabel',{'0','1'}, 'YTick',1:2,'YTickLabel',{'0','1'});
textStrings = strtrim(cellstr(num2str(C(:),'%d')));
[xText,yText] = meshgrid(1:2,1:2);
text(xText(:),yText(:),textStrings,'HorizontalAlignment','center', ...
     'Color','r','FontWeight','bold','FontSize',14);

% End of script
