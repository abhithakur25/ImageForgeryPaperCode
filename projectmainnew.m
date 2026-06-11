
clear all;
close all;
clc
TrainDatabasePath = uigetdir('F:\image forgery with siftand ransac\image forgery\train', 'Select training database path' );
DatabasePath = uigetdir('F:\image forgery with siftand ransac\image forgery\test', 'Select database path');
prompt = {'Enter  image name:'};
dlg_title = 'Input to face detection system';
num_lines= 1;
def = {'1'};

Image  = inputdlg(prompt,dlg_title,num_lines,def);
Image = strcat(DatabasePath,'\',char(Image),'.png');
original = imread(Image);
imshow(original)
original=imresize(original,[512 512]);
%original = imread('face.jpg');     % Matrox frame grabber + Pulnix camera
figure,imshow(original);
title('input image');

% explore color content using image viewer
%imview(rgb)
T = CreateDatabase(TrainDatabasePath);
Train_Number =4;
% smooth image (reduce noise/color variation)
rgb = imfilter(original,ones(3,3)/9);
%imview(rgb)
figure,imshow(rgb);
title('smooth image');
imwrite(rgb,'rgb.png');
original1 = rgb2gray(original);
% view image and RGB layers (nonuniform illumination)
figure(1), set(1,'position',[99 79 826 589])
subplot(2,2,1), subimage(rgb), title('fabric image'), axis off
subplot(2,2,2), map=gray(256); map(:,2:3)=0; subimage(rgb(:,:,1),map), title('red layer'), axis off
subplot(2,2,3), map=gray(256); map(:,[1 3])=0; subimage(rgb(:,:,2),map), title('green layer'), axis off
subplot(2,2,4), map=gray(256); map(:,1:2)=0; subimage(rgb(:,:,3),map), title('blue layer'), axis off
%rgb to ycbcr
 %i = imread('rgb.jpg');
 %newmap = rgb2ycbcr(i);
 %figure,imshow(newmap);

% RGB histograms (poor separability)
figure(1), set(1,'position',[452 68 560 420])
figure(2), set(2,'position',[16 269 560 420])
c='rgb';


f=rgb2ycbcr(original);
figure,imshow(f);
title('color converted image');


load my_regioncoordinates
%figure(3), imshow(rgb)
%for i=1:6
  %patch(roi(:,1,i),roi(:,2,i),'w','linestyle','none')
%end

% predefined regions (everything else suppressed)
mask = false([480 640 6]);
%for i=1:6
 % mask(:,:,i) = roipoly(rgb,roi(:,1,i),roi(:,2,i));
%end
bw=repmat(logical(sum(mask,3)),[1 1 3]);
im=rgb; % im(~bw)=nan;
figure(3), imshow(im)

nBins=5;
winSize=7;
nClass=6;

%Read Input Image
inImg = original;
%imshow(inImg);title('Input Image');

%Segmentation
outImg = colImgSeg(inImg, nBins, winSize, nClass);

%Displaying Output
figure;imshow(outImg);title('Segmentation Maps');
colormap('default');
% a = original;
% a = im2bw(a,0.5);
% siz = size(a);
% figure
% imshow(a)
% title('Input Image after conversion to Binary')
% fprintf('\n Click (1) to Perform Erosion')
% fprintf('\n Click (2) to Perform Dilation')
% fprintf('\n Click (3) to Perform Binary Opening')
% fprintf('\n Click (4) to Perform Binary Closing')
% fprintf('\n Click (5) to Subtract the Opening from the Input Image')
% fprintf('\n Click (6) to Subtract the Input Image from its Closing')
% fprintf('\n Click (7) to Remove isolated pixels (1''s surrounded by 0''s)')
% fprintf('\n Click (8) to Fill isolated interior pixels (0''s surrounded by 1''s)')
% fprintf('\n Click (9) to Leave only boundary pixels')
% fprintf('\n Click (10) to Shrink objects to points')
% fprintf('\n Click (11) to Make objects Thicker')
% fprintf('\n Click (12) to Make objects Thinner')
%                     
% choice = input('\n You select the Choice number : ');
% switch (choice)
% case 1
%     a = bwmorph(a,'erode');
% case 2
%     a = bwmorph(a,'dilate');
% case 3
%     a = bwmorph(a,'open');
% case 4
%     a = bwmorph(a,'close');
% case 5
%     a = bwmorph(a,'tophat');
% case 6
%     a = bwmorph(a,'bothat');
% case 7
%     a = bwmorph(a,'clean');
% case 8
%     a = bwmorph(a,'fill');
% case 9
%     a = bwmorph(a,'remove');
% case 10
%     a = bwmorph(a,'shrink',Inf);
% case 11
%     a = bwmorph(a,'thicken',Inf);
% case 12
%     a = bwmorph(a,'thin',Inf);
% otherwise
%     fprintf('\n Sorry! Wrong Choice')
% end
% figure
% imshow(a)
% title('Result Image')
% sav = input('\n Do you like to SAVE Result Image? (y/n) : ','s');
% if (sav == 'y')
%     fprintf('\n You choose to SAVE the Result Image')
%     naming = input('\n Type the name of the new image file (filename.ext) : ','s');
%     fprintf('\n Saving ...')
%     imwrite(a,'erode.jpg');
%     fprintf('\n The new file is called %s and it is saved in MATLAB working Directory',naming)
% else
%     fprintf('\n You choose NOT to SAVE the Result Image')
% end
cform = makecform('srgb2lab');
J = applycform(original,cform);
figure;imshow(J);
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
k1 = double(original1) + 80;

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
ext = 'jpg';
% img1 = imread([f '1.' ext]);
% img2 = imread([f '2.' ext]);
img1 = imread('1.jpg');
img2 = imread('3.jpg');
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
