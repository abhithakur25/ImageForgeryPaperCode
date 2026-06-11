function classes = likelihood2class(likelihoods)

%

[sample_n,class_n] = size(likelihoods);
maxs = (likelihoods==repmat(max(likelihoods,[],2),[1,class_n]));

classes=zeros(sample_n,1);
for i=1:sample_n
	classes(i) = find(maxs(i,:),1);
end

