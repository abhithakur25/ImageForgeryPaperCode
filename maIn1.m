clear all
close all
clc
tic
% READING INPUT IMAGE

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
img1=imread([pathname filename]);
figure,imshow(img1,[]);
title('original image');
% explore color content using image viewer
%imview(rgb)
T = CreateDatabase(TrainDatabasePath);
Train_Number =4;
% smooth image (reduce noise/color variation)
img1 = imfilter(img1,ones(3,3)/9);

for i=1:3
EQA=adapthisteq(img1(:,:,i));
cont(:,:,i)=imadjust(EQA);
end
figure,imshow(cont,[]);
title('color enhanced image');
img1=cont;
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
% st = circularstruct(lll); % lll varies from image to image
% OUT = imclose(frgd_rgns,st);
% OUT=imfill(OUT,'holes');
% figure
% imshow(OUT)
toc
% end
% %imview(rgb)
% figure,imshow(rgb);
% title('smooth image');
% imwrite(rgb,'rgb.png');
% original1 = rgb2gray(original);
% % view image and RGB layers (nonuniform illumination)
% figure(1), set(1,'position',[99 79 826 589])
% subplot(2,2,1), subimage(rgb), title('fabric image'), axis off
% subplot(2,2,2), map=gray(256); map(:,2:3)=0; subimage(rgb(:,:,1),map), title('red layer'), axis off
% subplot(2,2,3), map=gray(256); map(:,[1 3])=0; subimage(rgb(:,:,2),map), title('green layer'), axis off
% subplot(2,2,4), map=gray(256); map(:,1:2)=0; subimage(rgb(:,:,3),map), title('blue layer'), axis off
% %rgb to ycbcr
%  %i = imread('rgb.jpg');
%  %newmap = rgb2ycbcr(i);
%  %figure,imshow(newmap);
% 
% % RGB histograms (poor separability)
% figure(1), set(1,'position',[452 68 560 420])
% figure(2), set(2,'position',[16 269 560 420])
% c='rgb';
% 
% 
% f=rgb2ycbcr(original);
% figure,imshow(f);
% title('color converted image');
% 
% 
% load my_regioncoordinates
% %figure(3), imshow(rgb)
% %for i=1:6
%   %patch(roi(:,1,i),roi(:,2,i),'w','linestyle','none')
% %end
% 
% % predefined regions (everything else suppressed)
% mask = false([480 640 6]);
% %for i=1:6
%  % mask(:,:,i) = roipoly(rgb,roi(:,1,i),roi(:,2,i));
% %end
% bw=repmat(logical(sum(mask,3)),[1 1 3]);
% im=rgb; % im(~bw)=nan;
% figure(3), imshow(im)
% 
% nBins=5;
% winSize=7;
% nClass=6;
% 
% %Read Input Image
% inImg = original;
% %imshow(inImg);title('Input Image');
% 
% %Segmentation
% outImg = colImgSeg(inImg, nBins, winSize, nClass);
% 
% %Displaying Output
% figure;imshow(outImg);title('Segmentation Maps');
% colormap('default');
% % a = original;
% % a = im2bw(a,0.5);
% % siz = size(a);
% % figure
% % imshow(a)
% % title('Input Image after conversion to Binary')
% % fprintf('\n Click (1) to Perform Erosion')
% % fprintf('\n Click (2) to Perform Dilation')
% % fprintf('\n Click (3) to Perform Binary Opening')
% % fprintf('\n Click (4) to Perform Binary Closing')
% % fprintf('\n Click (5) to Subtract the Opening from the Input Image')
% % fprintf('\n Click (6) to Subtract the Input Image from its Closing')
% % fprintf('\n Click (7) to Remove isolated pixels (1''s surrounded by 0''s)')
% % fprintf('\n Click (8) to Fill isolated interior pixels (0''s surrounded by 1''s)')
% % fprintf('\n Click (9) to Leave only boundary pixels')
% % fprintf('\n Click (10) to Shrink objects to points')
% % fprintf('\n Click (11) to Make objects Thicker')
% % fprintf('\n Click (12) to Make objects Thinner')
% %                     
% % choice = input('\n You select the Choice number : ');
% % switch (choice)
% % case 1
% %     a = bwmorph(a,'erode');
% % case 2
% %     a = bwmorph(a,'dilate');
% % case 3
% %     a = bwmorph(a,'open');
% % case 4
% %     a = bwmorph(a,'close');
% % case 5
% %     a = bwmorph(a,'tophat');
% % case 6
% %     a = bwmorph(a,'bothat');
% % case 7
% %     a = bwmorph(a,'clean');
% % case 8
% %     a = bwmorph(a,'fill');
% % case 9
% %     a = bwmorph(a,'remove');
% % case 10
% %     a = bwmorph(a,'shrink',Inf);
% % case 11
% %     a = bwmorph(a,'thicken',Inf);
% % case 12
% %     a = bwmorph(a,'thin',Inf);
% % otherwise
% %     fprintf('\n Sorry! Wrong Choice')
% % end
% % figure
% % imshow(a)
% % title('Result Image')
% % sav = input('\n Do you like to SAVE Result Image? (y/n) : ','s');
% % if (sav == 'y')
% %     fprintf('\n You choose to SAVE the Result Image')
% %     naming = input('\n Type the name of the new image file (filename.ext) : ','s');
% %     fprintf('\n Saving ...')
% %     imwrite(a,'erode.jpg');
% %     fprintf('\n The new file is called %s and it is saved in MATLAB working Directory',naming)
% % else
% %     fprintf('\n You choose NOT to SAVE the Result Image')
% % end
% cform = makecform('srgb2lab');
% J = applycform(original,cform);
% figure;imshow(J);
% K=J(:,:,2);
% figure;imshow(K);
% L=graythresh(J(:,:,2));
% BW1=im2bw(J(:,:,2),L);
% figure;imshow(BW1);
% M=graythresh(J(:,:,3));
% figure;imshow(J(:,:,3));
% BW2=im2bw(J(:,:,3),M);
% figure;imshow(BW2);
% O=BW1.*BW2;
% % Bounding box
% k1 = double(original1) + 80;
% 
% bwim11=adaptivethreshold(k1,11,0.03,0);
% bwim1 = preprocessing(T);
% k3 = std(bwim11);
% k4 = skewness(bwim11);
% k5 = entropy(bwim11);
% disp(k5);
% k6 = kurtosis(bwim11);
% P=bwlabel(O,8);
% if (k5 > 0.5)
% BB=regionprops(P,'Boundingbox');
% BB1=struct2cell(BB);
% BB2=cell2mat(BB1);
% 
% [s1 s2]=size(BB2);
% mx=0;
% for k=3:4:s2-1
%     p=BB2(1,k)*BB2(1,k+1);
%     if p>mx & (BB2(1,k)/BB2(1,k+1))<1.8
%         mx=p;
%         j=k;
%     end
% end
% figure,imshow(original);
% hold on;
% rectangle('Position',[BB2(1,j-2),BB2(1,j-1),BB2(1,j),BB2(1,j+1)],'EdgeColor','r' );
% else 
%     figure,imshow(original);
% end
% 
% 
% [result,Euc_dist_min] =  knn(k6,bwim11);
% f = 'a';
% ext = 'jpg';
% % img1 = imread([f '1.' ext]);
% % img2 = imread([f '2.' ext]);
% img1 = imread('1.jpg');
% img2 = imread('3.jpg');
% img1 = imresize(img1,[449 677]);
% img2 = imresize(img2,[449 677]);
% % img3 = imread('b3.jpg');
% 
% img0 = imMosaic(img2,img1,1);
% % img0 = imMosaic(img1,img0,1);
% figure,imshow(img0)
% imwrite(img0,['mosaic_' f '.' ext],ext)
% 
% if (k5 > 0.5)
%     msgbox('FORGERY DETECTED','RESULT OF IMAGE FORGERY');
% else
%     msgbox('FORGERY NOT DETECTED','RESULT OF IMAGE FORGERY');
% end
