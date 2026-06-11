function bwim1 = preprocessing(T)
Train_Number = 4;
for i = 1 : Train_Number
k = histeq(T);

k1 = double(T) + 80;

bwim1=adaptivethreshold(k1,11,0.03,0);

end
end
