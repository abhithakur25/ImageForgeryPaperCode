%% ------------------ MORPHOLOGICAL POSTPROCESSING ------------------
ms = frgd_rgns;
ms = bwmorph(ms,'remove');
ms = imfill(ms,'holes');
ms = bwmorph(ms,'clean');

% Combine dilation stages once
se = strel('disk',15);
dil = imdilate(ms,se);
figure, imshow(dil); title('Forgery Detected Image');
imwrite(dil,'Forgerysegmentedimage.jpg');

%% ------------------ PERFORMANCE METRICS ------------------
k1 = double(dil) + 80;
bwim11 = adaptivethreshold(k1,11,0.03,0);
k5 = entropy(bwim11);

if (k5 > 0.01)
    msgbox('FORGERY DETECTED','RESULT OF IMAGE FORGERY');
else
    msgbox('FORGERY NOT DETECTED','RESULT OF IMAGE FORGERY');
end

%% ------------------ SVM-BASED ACCURACY EVALUATION ------------------
try
    load('Accuracy_Data.mat','Train_Feat','Train_Label');
catch
    warning('Accuracy_Data.mat not found — skipping SVM accuracy evaluation.');
    return;
end

fprintf('\nComputing SVM-based accuracy metrics...\n');

data   = double(Train_Feat);
labels = double(Train_Label(:));

% Ensure binary logical vector for classification
labels(labels~=0) = 1;
cv = cvpartition(labels,'HoldOut',0.25);
Xtrain = data(training(cv),:);
Ytrain = labels(training(cv));
Xtest  = data(test(cv),:);
Ytest  = labels(test(cv));

SVMModel = fitcsvm(Xtrain,Ytrain,'KernelFunction','linear','Standardize',true);
Ypred = predict(SVMModel,Xtest);

% Confusion matrix & metrics
C = confusionmat(Ytest,Ypred);
if size(C,1)==1
    C = [C(1) 0; 0 0];
end
TP = C(2,2); TN = C(1,1); FP = C(1,2); FN = C(2,1);

precision = TP/(TP+FP+eps);
recall    = TP/(TP+FN+eps);
F1        = 2*(precision*recall)/(precision+recall+eps);
accuracy  = (TP+TN)/sum(C(:));

fprintf('Precision: %.2f%%\n',precision*100);
fprintf('Recall:    %.2f%%\n',recall*100);
fprintf('F1 Score:  %.2f%%\n',F1*100);
fprintf('Accuracy:  %.2f%%\n',accuracy*100);

figure('Name','SVM Metrics');
bar([precision,recall,F1,accuracy]);
set(gca,'XTickLabel',{'Precision','Recall','F1','Accuracy'});
ylim([0 1]); ylabel('Score'); title('Performance Metrics');

% Confusion matrix plot
figure('Name','Confusion Matrix');
imagesc(C); axis equal tight; colorbar;
title('Confusion Matrix'); xlabel('Predicted'); ylabel('Actual');
textStrings = num2str(C(:)); textStrings = strtrim(cellstr(textStrings));
[xText,yText] = meshgrid(1:size(C,2),1:size(C,1));
text(xText(:),yText(:),textStrings(:),'HorizontalAlignment','center');
