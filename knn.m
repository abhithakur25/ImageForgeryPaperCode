function [result,Euc_dist_min]  = knn(k6,m);
Train_Number = 4;
for i = 1 : Train_Number
    k7 = kurtosis(i);
end
Euc_dist = [];
for i = 1 : Train_Number
    
    temp = ( norm( k7 - k6 ) )^2;
    Euc_dist = [Euc_dist temp];
end
[Euc_dist_min , Recognized_index] = min(Euc_dist);
result = strcat(int2str(Recognized_index),'.jpg');