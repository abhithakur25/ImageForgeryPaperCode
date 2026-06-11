clc;
clear all;
close all;
       
% READING INPUT IMAGE
[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
a=imread([pathname filename]);
% rgbim=imresize(I,[256 256]);
figure,imshow(a,[]);
title('original image');


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
