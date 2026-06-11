clear all;
close all;
clc
TrainDatabasePath = uigetdir('E:\MATLAB\Image Forgery\image forgery with siftand ransac\image forgery\train', 'Select training database path' );
DatabasePath = uigetdir('E:\MATLAB\Image Forgery\image forgery with siftand ransac\image forgery\test', 'Select database path');

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
original123=imread([pathname filename]);
original=original123;
 fprintf('\n Saving ...')
    imwrite(original,'original.jpg');
figure,imshow(original);
title('input image');
% explore color content using image viewer
%imview(rgb)
T = CreateDatabase(TrainDatabasePath);
Train_Number =4;
% smooth image (reduce noise/color variation)
rgb = imfilter(original,ones(3,3)/9);
figure,imshow(rgb);
title('smooth image');
imwrite(rgb,'rgb.jpg');


figure(), set(1,'position',[99 79 826 589])
subplot(2,2,1), subimage(rgb), title('fabric image'), axis off
subplot(2,2,2), map=gray(256); map(:,2:3)=0; subimage(rgb(:,:,1),map), title('red layer'), axis off
subplot(2,2,3), map=gray(256); map(:,[1 3])=0; subimage(rgb(:,:,2),map), title('green layer'), axis off
subplot(2,2,4), map=gray(256); map(:,1:2)=0; subimage(rgb(:,:,3),map), title('blue layer'), axis off
% RGB histograms (poor separability)
figure(1), set(1,'position',[452 68 560 420])
figure(2), set(2,'position',[16 269 560 420])
c='rgb';


f=rgb2ycbcr(rgb);
figure,imshow(f);
title('color converted image');
load my_regioncoordinates
% predefined regions (everything else suppressed)
mask = false([480 640 6]);
bw=repmat(logical(sum(mask,3)),[1 1 3]);
im=f; 
figure(3), imshow(im)

nBins=5;
winSize=7;
nClass=6;

%Read Input Image
inImg = im;
%imshow(inImg);title('Input Image');

%Segmentation
outImg = colImgSeg(inImg, nBins, winSize, nClass);

%Displaying Output
figure;imshow(outImg);title('Segmentation Maps');
colormap('default');
a = outImg;
% a = im2bw(a,0.5);
% siz = size(a);
% figure
% imshow(a)

%     a = bwmorph(a,'clean');
% 
% figure
% imshow(a)
title('Result of clean Image');
cform = makecform('srgb2lab');
J = applycform(original,cform);
figure;imshow(J);
img1 = imfilter(J,ones(3,3)/9);
img=rgb2gray(img1);
[row, col]=size(img);
[labels, num] = slicomex(img1,a);%numlabels is the same as number of superpixels
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
th_dist=0.25;
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
a=baseimgpts;
   
     a = bwmorph(a,'tophat');%figure,imshow(a),title('Subtract the Opening from the Input Image');
     
%      a = bwmorph(a,'clean');%figure,imshow(a),title('Remove isolated pixels (1''s surrounded by 0''s)');
%      
%      a = bwmorph(a,'fill');%figure,imshow(a),title('Fill isolated interior pixels (0''s surrounded by 1''s)');
%      
%       a = bwmorph(a,'remove');%figure,imshow(a),title('Leave only boundary pixels');
%       
%       a = bwmorph(a,'dilate');%figure,imshow(a),title('Dilate Result');
      baseimgpts=a;
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
   figure, imshow(frgd_rgns),title('frgd_rgns Final Result');
a=frgd_rgns;
a = bwmorph(a,'tophat');%figure,imshow(a),title('Subtract the Opening from the Input Image');
     
     a = bwmorph(a,'clean');%figure,imshow(a),title('Remove isolated pixels (1''s surrounded by 0''s)');
     
%      a = bwmorph(a,'fill');%figure,imshow(a),title('Fill isolated interior pixels (0''s surrounded by 1''s)');
     
      a = bwmorph(a,'remove');%figure,imshow(a),title('Leave only boundary pixels');
      
%       a = bwmorph(a,'dilate');%figure,imshow(a),title('Dilate Result');
frgd_rgns=a;
bwPicSize = size(frgd_rgns);

rgbPic = zeros(bwPicSize(1),bwPicSize(2),3);

rgbPic(frgd_rgns==1)=255;
rgbPic(:,:,2) = rgbPic(:,:,1);
rgbPic(:,:,3) = rgbPic(:,:,1);

rgbPic = im2uint8(rgbPic);
imshow(rgbPic)
% a=rgbPic;
% %     a = bwmorph(a,'tophat');%figure,imshow(a),title('Subtract the Opening from the Input Image');
%      
%      a = bwmorph(a,'clean');%figure,imshow(a),title('Remove isolated pixels (1''s surrounded by 0''s)');
%      
% %      a = bwmorph(a,'fill');%figure,imshow(a),title('Fill isolated interior pixels (0''s surrounded by 1''s)');
%      
%       a = bwmorph(a,'remove');%figure,imshow(a),title('Leave only boundary pixels');
%       
% %       a = bwmorph(a,'dilate');%figure,imshow(a),title('Dilate Result');

 fprintf('\n Saving ...')
    imwrite(a,'frgd_rgns_rgbPic.jpg');
    rgbPic=a;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cform = makecform('srgb2lab');
a = applycform(a,cform);
figure;imshow(a);

J=a;
K=J(:,:,2);
figure;imshow(K);
L=graythresh(J(:,:,2));
BW1=im2bw(J(:,:,2),L);
figure;imshow(BW1);
M=graythresh(J(:,:,3));
figure;imshow(J(:,:,3));
BW2=im2bw(J(:,:,3),M);
figure;imshow(BW2);
O=BW1.*BW2;
% Bounding box
k1 = double(original123) + 80;

bwim11=adaptivethreshold(k1,11,0.03,0);
bwim1 = preprocessing(T);
k3 = std(bwim11);
k4 = skewness(bwim11);
k5 = entropy(bwim11);
disp(k5);
k6 = kurtosis(bwim11);
P=bwlabel(O,8);
if (k5 > 0.5)
BB=regionprops(P,'Boundingbox');
BB1=struct2cell(BB);
BB2=cell2mat(BB1);

[s1 s2]=size(BB2);
mx=0;
for k=3:4:s2-1
    p=BB2(1,k)*BB2(1,k+1);
    if p>mx & (BB2(1,k)/BB2(1,k+1))<1.8
        mx=p;
        j=k;
    end
end
figure,imshow(original);
hold on;
rectangle('Position',[BB2(1,j-2),BB2(1,j-1),BB2(1,j),BB2(1,j+1)],'EdgeColor','r' );
else 
    figure,imshow(original);
end


[result,Euc_dist_min] =  knn(k6,bwim11);
f = 'a';
ext = 'png';
% img1 = imread([f '1.' ext]);
% img2 = imread([f '2.' ext]);
img1 = imread('frgd_rgns_rgbPic.jpg');
img2 = imread('original.jpg');
img1 = imresize(img1,[449 677]);
img2 = imresize(img2,[449 677]);
% img3 = imread('b3.jpg');

img0 = imMosaic(img2,img1,1);
% img0 = imMosaic(img1,img0,1);
figure,imshow(img0)
imwrite(img0,['mosaic_' f '.' ext],ext)

if (k5 > 0.5)
    msgbox('FORGERY DETECTED','RESULT OF IMAGE FORGERY');
else
    msgbox('FORGERY NOT DETECTED','RESULT OF IMAGE FORGERY');
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sav = input('\n Do you like to SAVE Result Image? (y/n) : ','s');
if (sav == 'y')
    fprintf('\n You choose to SAVE the Result Image')
    naming = input('\n Type the name of the new image file (filename.ext) : ','s');
    fprintf('\n Saving ...')
    imwrite(a,'erode.jpg');
    fprintf('\n The new file is called %s and it is saved in MATLAB working Directory',naming)
else
    fprintf('\n You choose NOT to SAVE the Result Image')
end