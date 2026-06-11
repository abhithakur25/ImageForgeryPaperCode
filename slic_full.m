 
%%%SLIC SEGMENTATION ALGORITHM%%%

function im=slic_full(cont,rgbim)
figure,imshow(rgbim),title('Original image');
figure,imshow(cont),title('reference image');
im=cont;
% [im rect]=imcrop(cont);
% %ini=round(ini);inj=round(inj);eni=round(eni);enj=round(enj);
% %im=img(ini:ini+eni,inj:inj+enj);
% rect=round(rect);
% rgbim=rgbim(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3),:);
% rgbim=imresize(rgbim,[250 250]);
% im=imresize(cont,[250 250]);
im=cont;
[h w d]=size(im);
C= makecform('srgb2lab');
labs =applycform(im,C);
labs=im2double(labs);
figure,imshow(labs),title('lab color space');
figure,imshow(rgbim),title(' rgb color space');

ll=labs(:,:,1);
aa=labs(:,:,2);
bb=labs(:,:,3);

%figure,imshow(ll);
%figure,imshow(aa);
%figure,imshow(bb);

%%% size of the image is 250x250%%
%%% total no. of pixels is 62500%%

%%% applying the 5x5 clustering
%cluster size cs=25;
S=5;
%totally 25 pixels in one (5x5)cluster
cs=S*S;                          
%totally 2500 cluster centers(superpixels)
K=(h*w)/cs;





checki=zeros(1,K);
checkj=zeros(1,K);
ci=zeros(1,K);
cj=zeros(1,K);

centg=zeros(3,3);
count=1;
for i=3:S:h
   for j=3:S:w
        %Lowest gradient computation for cluster centers
        %i=8;j=8; 
        ha=i-1;hb=i+1;wa=j-1; wb=j+1;
        centg=gradient(labs(ha:hb,wa:wb));
        minval=min(min(centg));
        [rc,cc]=find(centg==minval);
        
        checki(count)=i;
        checkj(count)=j;
        
        % to find the center point exactly around 320x320 in stead of 3x3
        % gradient
        if(rc==1)
            rea=-1;
        elseif (rc==2)
            rea=0;
        else
            rea=1;
        end
        
        if(cc==1)
            cea=-1;
        elseif (cc==2)
            cea=0;
        else
            cea=1;
        end
        
        %for storing the center values on 320x320 space
        ci(count)=i+rea;
        cj(count)=j+cea;
        count=count+1;
        
    end
end

%%% ---- 2Sx2S process-----------------------------------------------------

sd=2*S+1;
l=-1*ones(sd,sd,K);
d=100000000*ones(sd,sd,K);
m=10;

checksi=zeros(sd,K);
checksj=zeros(sd,K);
nnnci=zeros(1,K);
nnncj=zeros(1,K);

for C=1:K
    mci=ci(C);
    mcj=cj(C);
    cntt=1;
    for si=mci-S:mci+S
        if (si<1 || si>h)
           continue;
        end
        cnt=1;
        for sj=mcj-S:mcj+S
            if (sj<1 || sj>w)
                continue;
            end
            
            nnnci(1,C)=mci;
            nnncj(1,C)=mcj;
            
            lld=abs(ll(mci,mcj)-ll(si,sj))^2;
            aad=abs(aa(mci,mcj)-aa(si,sj))^2;
            bbd=abs(bb(mci,mcj)-bb(si,sj))^2;
            dc=sqrt(lld+aad+bbd);
            
            ds=sqrt(((mcj-sj)^2)+((mci-si)^2));
            
            D=sqrt(dc^2+((ds/S)^2*m^2));
            
            checksi(cntt,C)=si;
            checksj(cnt,C)=sj;
            
            if D<d(cntt,cnt,C)
                %distance bw center to other 2s x 2s pixels
                d(cntt,cnt,C)=D;
                l(cntt,cnt,C)=C;
           end
            cnt=cnt+1;
       end
        cntt=cntt+1;
        
    end
end


% to find new cluster using mean operation
finalci=[];
finalcj=[];
ffci=[];
ffcj=[];
lsum=[];
llsum=[];
eucdi=zeros(1,C);
eucdj=zeros(1,C);

for C=1:K
    ss=[];
    Ciini=checksi(1,C);
    Ciend=max(checksi(:,C));
    Cjini=checksj(1,C);
    Cjend=max(checksj(:,C));
    
    for xx=Ciini:Ciend
        for yy=Cjini:Cjend
             ss=[ss; labs(xx,yy)];
        end 
    end
    
   % for taking the mean value:
   ss=sort(ss);
   bin=ceil((size(ss,1)/2));
   findd=ss(bin,1);
   [finalcii finalcjj]=find(labs(Ciini:Ciend,Cjini:Cjend)==findd);
   % if more than one element matched then to take the first one value
   
   nciini=Ciini+finalcii(1)-1;
   ncjini=Cjini+finalcjj(1)-1;
   
   %to find the center point within 2Sx2S block
   finalci=[finalci finalcii(1)];
   finalcj=[finalcj finalcjj(1)];
   
   %to find the exact center point
   ffci=[ffci nciini];
   ffcj=[ffcj ncjini];
   
    %%finding euclidean dist
    eucdi(C)=pdist2(ffci(C),nnnci(C));
    eucdj(C)=pdist2(ffcj(C),nnncj(C)); 
    
end

%itrsh=graythresh(eucdi);
%jtrsh=graythresh(eucdj);
%it=im2bw(ecudi,itrsh);
%jt=im2bw(ecudj,jtrsh);

% if it=0 converges else once again repeat the center finding procedure
   
imcc=im;
outtp=labs;
for k=1:K
outtp(ffci(k),ffcj(k),:)=255;
imcc(ffci(k),ffcj(k),:)=255;
rgbim(ffci(k),ffcj(k),:)=255;

end

figure,imshow(outtp),title('lab space with cluster centers');
figure,imshow(imcc),title('image with cluster centers ');
figure,imshow(rgbim),title('image with cluster centers ');
end