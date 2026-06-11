% mift_sift_demo demonstrate MIFT code
% Created by: Xiaojie Guo
% Affiliate: School of Computer Science and Technology, Tianjin Univ.,
%            Tianjin, 300072, China
% Reference: [1] MIFT: A Mirror Reflection Invariant Descriptor,        
%                Xiaojie Guo, Xiaochun Cao, Jiawan Zhang and Xuewei Li,
%                ACCV, 2009.
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This implementation is based on the codes written by Andrea Vadaldi 
% UCLA Vision Lab - Department of Computer Science
% Copyright (c) 2006 The Regents of the University of California.
% All Rights Reserved.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
filename1='img1.bmp';
filename2='img3.bmp';
I1=imreadbw(filename1) ; 
I2=imreadbw(filename2) ;

II1=imread(filename1);
II2=imread(filename2);
I1=I1-min(I1(:)) ;
I1=I1/max(I1(:)) ;
I2=I2-min(I2(:)) ;
I2=I2/max(I2(:)) ;

fprintf('Computing frames and descriptors.\n') ;
[frames1,descr1,gss1,dogss1,ori1,tau1] = mift_sift( I1, 'Verbosity', 0 ) ;
[frames2,descr2,gss2,dogss2,ori2,tau2] = mift_sift( I2, 'Verbosity', 0 ) ;
fprintf('Finding correspondences between the images.\n') ;
cols1 = size(I1,2);
j=0;
count=0;
match=[];
for i = 1 : size(descr1,2)
       dotprods = descr1(:,i)' * descr2;      
       [vals,indx] = sort(acos(dotprods)); 
       k=2;
       distRatio = 0.6; 
       if(vals(1)<distRatio*vals(k))
        match(i)=indx(1);
           else
           match(i)=0;
         end
       while(true)
           if(vals(1)<distRatio*vals(k))
               match(i)=indx(1);
           else
               match(i)=0;
           end
           if((match(i)~=0)|allEleUnequal(frames2(:,indx(1))==frames2(:,indx(k))))
               break;
           end
           k=k+1;
       end
       if(match(i)==0)
           thrA=0.9;
           addCri=descr2(indx(k))'*descr2(indx(1));
           if(addCri>thrA)
               match(i)=indx(1);
           end
       end

end
fprintf('Showing results.\n') ;
im3 = appendimages(II1,II2);
figure('Position', [100 100 size(im3,2) size(im3,1)]);
imshow(im3);
hold on;
 colour=['c','r','g','y','w','b'];
d=[];
rev=0;
same=0;
TauS=0;
TauR=0;
for i = 1: size(descr1,2)
  if (match(i) > 0)
      count=count+1;
     line([frames1(1,i)  frames2(1,match(i))+cols1], ...
         [frames1(2,i)  frames2(2,match(i))], 'Color',colour(mod(count,6)+1));
     if(ori1(1,i)~=ori2(1,match(i)))
         rev=rev+1;
          if(tau1(1,i)||tau2(1,match(i)))
             TauS=TauS+1;
         end
     else
         same=same+1;
         if(tau1(1,i)||tau2(1,match(i)))
             TauR=TauR+1;
         end
     end
  end
end

fprintf('Done.\n') ;

