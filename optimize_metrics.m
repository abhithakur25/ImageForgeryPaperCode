function optimize_metrics()
% OPTIMIZE_METRICS  Train and compare several classifiers on Accuracy_Data.mat
% using stratified 5-fold cross-validation, then report the best model across
% precision, recall, F1, ROC-AUC, training accuracy, validation accuracy and
% overall accuracy. Picks the positive class = 1 (forged).
%
% Produces: best_model.mat, metrics_summary.csv, figures saved as PNG.

clc;
fprintf('================ TREE/FORGERY DETECTION : METRIC OPTIMIZATION ================\n');

%% ---- Load data ----
S = load('Accuracy_Data.mat');
X = double(S.Train_Feat);
y = double(S.Train_Label(:));
y(y~=0) = 1;                      % ensure binary {0,1}
posClass = 1;                    % minority / forged class

fprintf('Samples: %d   Features: %d   (class0=%d, class1=%d)\n', ...
    size(X,1), size(X,2), sum(y==0), sum(y==1));

% Clean any non-finite just in case
X(~isfinite(X)) = 0;

%% ---- Reproducibility & stratified CV partition ----
rng(7,'twister');
K = 5;
cv = cvpartition(y,'KFold',K);   % stratified by class for classification labels

%% ---- Candidate models (as builder function handles) ----
models = {};
models(end+1,:) = {'Linear SVM',      @(Xtr,ytr) fitcsvm(Xtr,ytr,'KernelFunction','linear','Standardize',true,'ClassNames',[0;1])};
models(end+1,:) = {'RBF SVM (auto)',  @(Xtr,ytr) fitcsvm(Xtr,ytr,'KernelFunction','rbf','KernelScale','auto','Standardize',true,'Cost',[0 1;4 0],'ClassNames',[0;1])};
models(end+1,:) = {'Bagged Trees (RF)',@(Xtr,ytr) fitcensemble(Xtr,ytr,'Method','Bag','NumLearningCycles',200,'Learners',templateTree('MinLeafSize',1),'ClassNames',[0;1])};
models(end+1,:) = {'RUSBoost',        @(Xtr,ytr) fitcensemble(Xtr,ytr,'Method','RUSBoost','NumLearningCycles',200,'LearnRate',0.1,'Learners',templateTree('MaxNumSplits',20),'ClassNames',[0;1])};
models(end+1,:) = {'AdaBoostM1',      @(Xtr,ytr) fitcensemble(Xtr,ytr,'Method','AdaBoostM1','NumLearningCycles',200,'LearnRate',0.1,'ClassNames',[0;1])};
models(end+1,:) = {'Weighted kNN',    @(Xtr,ytr) fitcknn(Xtr,ytr,'NumNeighbors',5,'DistanceWeight','squaredinverse','Standardize',true,'ClassNames',[0;1])};

nM = size(models,1);
results = struct('name',{},'precision',{},'recall',{},'F1',{},'AUC',{}, ...
                 'valAcc',{},'trainAcc',{},'TP',{},'FP',{},'TN',{},'FN',{});

%% ---- Evaluate each model with the SAME CV folds ----
for m = 1:nM
    name    = models{m,1};
    builder = models{m,2};

    yhat   = zeros(size(y));
    score1 = zeros(size(y));     % score for positive class
    trainAccFolds = zeros(K,1);

    for f = 1:K
        tr = training(cv,f);
        te = test(cv,f);
        mdl = builder(X(tr,:), y(tr));

        % training accuracy on this fold's training data
        trainAccFolds(f) = mean(predict(mdl, X(tr,:)) == y(tr));

        % validation predictions + scores
        [pl, sc] = predict(mdl, X(te,:));
        yhat(te) = pl;
        % locate positive-class score column
        cn = mdl.ClassNames;
        pcol = find(cn==posClass,1);
        score1(te) = sc(:,pcol);
    end

    % Confusion counts (positive = class 1)
    TP = sum(yhat==1 & y==1);
    FP = sum(yhat==1 & y==0);
    TN = sum(yhat==0 & y==0);
    FN = sum(yhat==0 & y==1);

    precision = TP/(TP+FP+eps);
    recall    = TP/(TP+FN+eps);
    F1        = 2*precision*recall/(precision+recall+eps);
    valAcc    = (TP+TN)/numel(y);
    trainAcc  = mean(trainAccFolds);

    % ROC-AUC from cross-validated scores
    try
        [~,~,~,AUC] = perfcurve(y, score1, posClass);
    catch
        AUC = NaN;
    end

    results(m) = struct('name',name,'precision',precision,'recall',recall, ...
        'F1',F1,'AUC',AUC,'valAcc',valAcc,'trainAcc',trainAcc, ...
        'TP',TP,'FP',FP,'TN',TN,'FN',FN);

    fprintf('%-18s | P=%.3f R=%.3f F1=%.3f AUC=%.3f | valAcc=%.3f trainAcc=%.3f\n', ...
        name, precision, recall, F1, AUC, valAcc, trainAcc);
end

%% ---- Pick best model: rank by (F1 + AUC + valAcc) ----
score = arrayfun(@(r) r.F1 + r.AUC + r.valAcc, results);
[~,best] = max(score);
B = results(best);

fprintf('\n================ BEST MODEL: %s ================\n', B.name);
fprintf('Precision        : %.2f%%\n', B.precision*100);
fprintf('Recall           : %.2f%%\n', B.recall*100);
fprintf('F1 Score         : %.2f%%\n', B.F1*100);
fprintf('ROC-AUC          : %.4f\n',   B.AUC);
fprintf('Validation Acc   : %.2f%%\n', B.valAcc*100);
fprintf('Training Acc      : %.2f%%\n', B.trainAcc*100);
fprintf('Overall Accuracy : %.2f%%\n', B.valAcc*100);
fprintf('Confusion [TP FP TN FN] = [%d %d %d %d]\n', B.TP,B.FP,B.TN,B.FN);

%% ---- Refit best model on ALL data, save ----
bestBuilder = models{best,2};
finalModel = bestBuilder(X,y);
save('best_model.mat','finalModel','B','results');

%% ---- Write CSV summary ----
fid = fopen('metrics_summary.csv','w');
fprintf(fid,'Model,Precision,Recall,F1,AUC,ValAccuracy,TrainAccuracy,TP,FP,TN,FN\n');
for m=1:nM
    r = results(m);
    fprintf(fid,'%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n', ...
        r.name,r.precision,r.recall,r.F1,r.AUC,r.valAcc,r.trainAcc,r.TP,r.FP,r.TN,r.FN);
end
fclose(fid);

%% ---- Plots (saved headless) ----
% Bar chart of best metrics
f1 = figure('Visible','off','Position',[100 100 700 500]);
vals = [B.precision B.recall B.F1 B.AUC B.valAcc B.trainAcc];
bar(vals); ylim([0 1]);
set(gca,'XTickLabel',{'Precision','Recall','F1','AUC','ValAcc','TrainAcc'});
ylabel('Score'); title(sprintf('Best Model: %s', B.name)); grid on;
saveas(f1,'best_metrics_bar.png'); close(f1);

% ROC curve of best model (from CV scores recomputed)
yhat = zeros(size(y)); score1 = zeros(size(y));
for f=1:K
    tr=training(cv,f); te=test(cv,f);
    mdl=bestBuilder(X(tr,:),y(tr));
    [pl,sc]=predict(mdl,X(te,:));
    pcol=find(mdl.ClassNames==posClass,1);
    yhat(te)=pl; score1(te)=sc(:,pcol);
end
[Xr,Yr,~,AUC] = perfcurve(y,score1,posClass);
f2 = figure('Visible','off','Position',[100 100 600 500]);
plot(Xr,Yr,'LineWidth',2); hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)', AUC)); grid on;
saveas(f2,'best_roc_curve.png'); close(f2);

fprintf('\nSaved: best_model.mat, metrics_summary.csv, best_metrics_bar.png, best_roc_curve.png\n');
fprintf('DONE_OPTIMIZE\n');
end
