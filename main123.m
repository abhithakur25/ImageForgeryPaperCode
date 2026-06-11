clear all
close all
clc
tic
% READING INPUT IMAGE

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
original=imread([pathname filename]);
figure,imshow(original,[]);
title('original image');
%Read Input Image
a = original;
a = im2bw(a,0.5);
siz = size(a);
    a = bwmorph(a,'remove');
figure
imshow(a)
title('Result Image')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img1 = imfilter(original,ones(3,3)/9);

for i=1:3
EQA=adapthisteq(img1(:,:,i));
cont(:,:,i)=imadjust(EQA);
end
figure,imshow(cont,[]);
title('color enhanced image');
img1=cont;
[labels, num] = slicomex(img1,a);%numlabels is the same as number of superpixels
labels=labels+1;
immm=drawregionboundaries(labels,img1,[255 0 0]);
figure; imshow(immm)
% imcc=slic_full(immm,img1)

%%

tot_fpts=0;
desc=cell(1,num);
loc=cell(1,num);
for i=1:num
    mask=labels~=i;
    imageFile=a;
        imageFile(mask)=0;
    [locs,descriptors] = mift_sift( imageFile, 'Verbosity', 0 ) ;
    clear imageFile
    desc{1,i}=descriptors';
    loc{1,i}=locs';
end

%% matching between patches
th_dist=0.15;
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
                    [index, distance] = knnsearch(descB, descA,'K', 3, 'NSMethod', 'kdtree', 'distance', 'euclidean') ;
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
figure
imshow(baseimgpts)
imgR=double(img1(:,:,1));
imgG=double(img1(:,:,2));
imgB=double(img1(:,:,3));
frgd_rgns=zeros(row,col);
if ~isempty(matched_pts)
matched_pts=round(sortrows(matched_pts,6));
nL=numel(unique(matched_pts(:,6)));
matched_pts(:,[1 2 3 4])=matched_pts(:,[2 1 4 3]);
[IDX,C] = kmeans(matched_pts(:,6), nL, 'EmptyAction','singleton');
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
    baseimgpts=frgd_rgns;
    end
    clear G
    clear measurementsRed
    clear measurementsGreen
    clear measurementsBlue
end 
end
figure
imshow(frgd_rgns)
sav = input('\n Do you like to SAVE Result Image? (y/n) : ','s');
if (sav == 'y')
    fprintf('\n You choose to SAVE the Result Image')
    naming = input('\n Type the name of the new image file (filename.ext) : ','s');
    fprintf('\n Saving ...')
    imwrite(a,'erode.png');
    fprintf('\n The new file is called %s and it is saved in MATLAB working Directory',naming)
else
    fprintf('\n You choose NOT to SAVE the Result Image')
end











