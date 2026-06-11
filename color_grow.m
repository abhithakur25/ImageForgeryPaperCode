function J=color_grow(Irgb,Isus,initPos1,initPos2,TR,TG,TB)


[nRow,nCol] = size(Isus);

if initPos1(1) < 1 || initPos1(2) < 1 ||...
   initPos1(1) > nRow || initPos1(2) > nCol
    error('Initial position out of bounds, please try again!')
end
if initPos2(1) < 1 || initPos2(2) < 1 ||...
   initPos2(1) > nRow || initPos2(2) > nCol
    error('Initial position out of bounds, please try again!')
end

J=Isus;
% maxxx=1.6;
% minnn=0.4;
% count=1000;
% % maxxx=1.45;
% % minnn=0.55;
% % count=785;
% % % maxxx=1.43;
% % % minnn=0.57;
% % % count=1500;
% % % % maxxx=1.38;
% % % % minnn=0.4;
% % % % count=1500;
% maxxx=1.38;
% minnn=0.38;
% count=1450;
maxxx=1.4;
minnn=0.45;
count=1500;
queue1 = [initPos1(1), initPos1(2)];
queue2 = [initPos2(1), initPos2(2)];
%%% START OF REGION GROWING ALGORITHM
nlx=1;
while size(queue1, 1)
  % the first queue position determines the new values
  xv1 = queue1(1,1);
  yv1 = queue1(1,2);
  xv2 = queue2(1,1);
  yv2 = queue2(1,2);
 
  % .. and delete the first queue position
  queue1(1,:) = [];
  queue2(1,:) = [];  
  % check the neighbors for the current position
  for i = -1:1
    for j = -1:1
%       for k = -1:1
            
        if xv1+i > 0  &&  xv1+i <= nRow &&...          % within the x-bounds?
           yv1+j > 0  &&  yv1+j <= nCol &&...          % within the y-bounds?          
           xv2+i > 0  &&  xv2+i <= nRow &&...          % within the x-bounds?
           yv2+j > 0  &&  yv2+j <= nCol &&...          % within the y-bounds?
           J(xv1+i, yv1+j)==0  &&...          % pixelposition already set?
           J(xv2+i, yv2+j)==0  &&...          % pixelposition already set?
          Irgb(xv1+i,yv1+j,1)  <= maxxx*TR  &&...   % within distance?
          Irgb(xv2+i,yv2+j,1)  <= maxxx*TR   &&...   % within distance?
          Irgb(xv1+i,yv1+j,2)  <= maxxx*TG   &&...   % within distance?
          Irgb(xv2+i,yv2+j,2)  <= maxxx*TG   &&...   % within distance?
          Irgb(xv1+i,yv1+j,3)  <= maxxx*TB   &&...   % within distance?
          Irgb(xv2+i,yv2+j,3)  <= maxxx*TB   &&...   % within distance?
          Irgb(xv1+i,yv1+j,1)  >= minnn*TR   &&...   % within distance?
          Irgb(xv2+i,yv2+j,1)  >= minnn*TR   &&...   % within distance?
          Irgb(xv1+i,yv1+j,2)  >= minnn*TG   &&...   % within distance?
          Irgb(xv2+i,yv2+j,2)  >= minnn*TG   &&...   % within distance?
          Irgb(xv1+i,yv1+j,3)  >= minnn*TB   &&...   % within distance?
          Irgb(xv2+i,yv2+j,3)  >= minnn*TB   &&...
          mean(Irgb(xv2+i,yv2+j,:)) >= (minnn+0.1)*mean([TR,TG,TB])  &&...
          mean(Irgb(xv2+i,yv2+j,:)) <= (maxxx-0.1)*mean([TR,TG,TB])   &&...
      any([i, j])            % i/j/k of (0/0/0) is redundant!
%           abs(sqrt(((xv1+i-initPos1(1))^2)+((yv1+j-initPos1(2))^2)))  <100  &&...
%           abs(sqrt(((xv2+i-initPos2(1))^2)+((yv2+j-initPos2(2))^2)))  <100  
      % current pixel is true, if all properties are fullfilled
           J(xv1+i, yv1+j) = 255; 
           J(xv2+i, yv2+j) = 255; 
           % add the current pixel to the computation queue (recursive)
           queue1(end+1,:) = [xv1+i, yv1+j];
           queue2(end+1,:) = [xv2+i, yv2+j];

%            if tfMean
%                regVal = mean(mean(cIM(J > 0))); % --> slow!
%            end

  %      end        
        nlx=nlx+1;
        if nlx==count
            break
        end
        end
        if nlx==count
            break
        end
    end  
    if nlx==count
            break
    end
  end
  if nlx==count
            break
  end
end
%%% END OF REGION GROWING ALGORITHM