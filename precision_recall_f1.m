close all;
clear all;
clc;
%% Input  forg Image from Database

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
forg=imread([pathname filename]);
[NUMROWS,NUMCOLS] = size(forg)
b=forg;
figure,imshow(forg);
title('input forg image');


%% Input original Image from Database

[filename pathname]=uigetfile( {'*.jpg'; '*.bmp';'*.tif';'*.png'});
original=imread([pathname filename]);
% original=imresize(original,[NUMROWS NUMCOLS]);
[NUMROWS,NUMCOLS] = size(original)
figure,imshow(original);
title('input image');

a=original;
[m,n]=size(original);
p=0;
q=0;
p1=0;
q1=0;
for i=1:m
    for j=1:n
        if b(i,j)==1
            p=p+1;
        end
        if b(i,j)==0
            p1=p1+1;
        end
        if (a(i,j)==0)||(b(i,j)==0)
                q=q+1;
        end
            if (a(i,j)~=0)||(b(i,j)~=0)
                 q1=q1+1;
            end
         end
end
precision=((q-q1)/(p+p1))*100
r=0;
s=0;
r1=0;
s1=0;
for i=1:m
    for j=1:n
        if a(i,j)==1
            r=r+1;
        end
        if a(i,j)==0
            r1=r1+1;
        end
            if (a(i,j)==1)||(b(i,j)==1)
                s=s+1;
            end
             if (a(i,j)==0)||(b(i,j)==0)
                 s1=s1+1;
            end
        end
end
recall=((s-s1)/(r+r1))*100
f=2*((precision*recall)/(precision+recall))