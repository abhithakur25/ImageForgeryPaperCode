clear all;
close all;
clc
TrainDatabasePath = uigetdir('E:\MATLAB\Image Forgery\image forgery with siftand ransac\image forgery\train', 'Select training database path' );
DatabasePath = uigetdir('E:\MATLAB\Image Forgery\image forgery with siftand ransac\image forgery\test', 'Select database path');

prompt = {'Enter  image name:'};
dlg_title = 'Input to face detection system';
num_lines= 1;
def = {'5'};

Image  = inputdlg(prompt,dlg_title,num_lines,def);
Image = strcat(DatabasePath,'\',char(Image),'.png');
original123 = imread(Image);
original=original123;
figure,imshow(original);
title('input image');
colormap('default');
a = original;
a = im2bw(a,0.5);
siz = size(a);

    a1 = bwmorph(a,'erode');figure,imshow(a1),title('Perform Erosion');

    a2 = bwmorph(a,'dilate');figure,imshow(a2),title('Perform Dilation');

    a3 = bwmorph(a,'open');figure,imshow(a3),title('Perform Binary Opening');

    a4 = bwmorph(a,'close');figure,imshow(a4),title('Perform Binary Closing');

    a5 = bwmorph(a,'tophat');figure,imshow(a5),title('Subtract the Opening from the Input Image');

    a6 = bwmorph(a,'bothat');figure,imshow(a6),title('Subtract the Input Image from its Closing');

    a7 = bwmorph(a,'clean');figure,imshow(a7),title('Remove isolated pixels (1''s surrounded by 0''s)');

    a8 = bwmorph(a,'fill');figure,imshow(a8),title('Fill isolated interior pixels (0''s surrounded by 1''s)');

    a9 = bwmorph(a,'remove');figure,imshow(a9),title('Leave only boundary pixels');

    a10 = bwmorph(a,'shrink',Inf);figure,imshow(a10),title('Shrink objects to points');

    a11 = bwmorph(a,'thicken',Inf);figure,imshow(a11),title('Make objects Thicker');

    a12 = bwmorph(a,'thin',Inf);figure,imshow(a12),title('Make objects Thinner');
    
    img1 = imfilter(a7,ones(3,3)/9);
% 
% for i=1:3
% EQA=adapthisteq(img1(:,:,i));
% cont(:,:,i)=imadjust(EQA);
% end
% figure,imshow(cont,[]);
% title('color enhanced image');
% img1=cont;
[labels, num] = slicomex(img1,a7);%numlabels is the same as number of superpixels
labels=labels+1;
immm=drawregionboundaries(labels,img1,[255 0 0]);
figure; imshow(immm)

%     a1 = bwmorph(a,'erode');figure,imshow(a1),title('Perform Erosion');
% 
%     a2 = bwmorph(a,'dilate');figure,imshow(a2),title('Perform Dilation');
% 
%     a3 = bwmorph(a,'open');figure,imshow(a3),title('Perform Binary Opening');
% 
%     a4 = bwmorph(a,'close');figure,imshow(a4),title('Perform Binary Closing');
% 
%     a5 = bwmorph(a,'tophat');figure,imshow(a5),title('Subtract the Opening from the Input Image');
% 
%     a6 = bwmorph(a,'bothat');figure,imshow(a6),title('Subtract the Input Image from its Closing');
% 
%     a7 = bwmorph(a,'clean');figure,imshow(a7),title('Remove isolated pixels (1''s surrounded by 0''s)');
% 
%     a8 = bwmorph(a,'fill');figure,imshow(a8),title('Fill isolated interior pixels (0''s surrounded by 1''s)');
% 
%     a9 = bwmorph(a,'remove');figure,imshow(a9),title('Leave only boundary pixels');
% 
%     a10 = bwmorph(a,'shrink',Inf);figure,imshow(a10),title('Shrink objects to points');
% 
%     a11 = bwmorph(a,'thicken',Inf);figure,imshow(a11),title('Make objects Thicker');
% 
%     a12 = bwmorph(a,'thin',Inf);figure,imshow(a12),title('Make objects Thinner');