clear all;
close all;
clc;
clf;


original = imread('face.jpg');     % Matrox frame grabber + Pulnix camera
figure,imshow(original);
title('input image');

% explore color content using image viewer
%imview(rgb)

% smooth image (reduce noise/color variation)
rgb1 = imfilter(original,ones(3,3)/9);
%imview(rgb)
figure,imshow(rgb1);
title('smooth image');
imwrite(rgb1,'rgb1.jpg');

% view image and RGB layers (nonuniform illumination)
figure(1), set(1,'position',[99 79 826 589])
subplot(2,2,1), subimage(rgb), title('fabric image'), axis off
subplot(2,2,2), map=gray(256); map(:,2:3)=0; subimage(rgb(:,:,1),map), title('red layer'), axis off
subplot(2,2,3), map=gray(256); map(:,[1 3])=0; subimage(rgb(:,:,2),map), title('green layer'), axis off
subplot(2,2,4), map=gray(256); map(:,1:2)=0; subimage(rgb(:,:,3),map), title('blue layer'), axis off rgb to ycbcr
 i = imread('rgb.jpg');
 newmap = rgb2ycbcr(i);
 figure,imshow(newmap);

% RGB histograms (poor separability)
figure(1), set(1,'position',[452 68 560 420])
figure(2), set(2,'position',[16 269 560 420])
c='rgb';
%for i=1:3
%  n=hist(reshape(double(rgb(:,:,i)),[480*640 1]),0.5:256);
 %  n=hist( rgb(:,:,i),[480*640 1],0.5:256);
  %line(0:255,n,'color',c(i))
%end
%axis tight, xlim([0 255]), box on
%xlabel intensity, ylabel population, title histograms

% convert image to L*a*b* color space (transform)
%cform = makecform('srgb2lab');
%lab = applycform(rgb,cform);
f=rgb2ycbcr(original);
figure,imshow(f);
title('color converted image');

% view components (note illumination free)
%figure(1), figure(2)
%subplot(2,2,1), subimage(rgb), title('fabric image'), axis off
%subplot(2,2,2), subimage(lab(:,:,1)), title('L* layer'), axis off
%subplot(2,2,3), map=gray(256); map(:,3)=0; map(:,2)=map(end:-1:1,2); subimage(lab(:,:,2),map), title('a* layer'), axis off
%subplot(2,2,4), map=gray(256); map(:,3)=map(end:-1:1,3); subimage(lab(:,:,3),map), title('b* layer'), axis off

% select polygon region of interest
%figure, imshow(rgb), roipoly

% predefined regions for 6 different colors present
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
a = original;
a = im2bw(a,0.5);
siz = size(a);
figure
imshow(a)
title('Input Image after conversion to Binary')
fprintf('\n Click (1) to Perform Erosion')
fprintf('\n Click (2) to Perform Dilation')
fprintf('\n Click (3) to Perform Binary Opening')
fprintf('\n Click (4) to Perform Binary Closing')
fprintf('\n Click (5) to Subtract the Opening from the Input Image')
fprintf('\n Click (6) to Subtract the Input Image from its Closing')
fprintf('\n Click (7) to Remove isolated pixels (1''s surrounded by 0''s)')
fprintf('\n Click (8) to Fill isolated interior pixels (0''s surrounded by 1''s)')
fprintf('\n Click (9) to Leave only boundary pixels')
fprintf('\n Click (10) to Shrink objects to points')
fprintf('\n Click (11) to Make objects Thicker')
fprintf('\n Click (12) to Make objects Thinner')
                    
choice = input('\n You select the Choice number : ');
switch (choice)
case 1
    a = bwmorph(a,'erode');
case 2
    a = bwmorph(a,'dilate');
case 3
    a = bwmorph(a,'open');
case 4
    a = bwmorph(a,'close');
case 5
    a = bwmorph(a,'tophat');
case 6
    a = bwmorph(a,'bothat');
case 7
    a = bwmorph(a,'clean');
case 8
    a = bwmorph(a,'fill');
case 9
    a = bwmorph(a,'remove');
case 10
    a = bwmorph(a,'shrink',Inf);
case 11
    a = bwmorph(a,'thicken',Inf);
case 12
    a = bwmorph(a,'thin',Inf);
otherwise
    fprintf('\n Sorry! Wrong Choice')
end
figure
imshow(a)
title('Result Image')
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
clear


    
