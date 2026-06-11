function optimize_dl()
% OPTIMIZE_DL  Deep-learning classifier for tree/forgery detection metrics.
% Uses only base MATLAB + Deep Learning Toolbox (no Statistics toolbox).
% Stratified k-fold CV, class-weighted loss for imbalance, ensemble over seeds.
% Reports precision, recall, F1, ROC-AUC, training acc, validation acc, overall acc.

clc;
fprintf('============ TREE/FORGERY DETECTION : DEEP-LEARNING OPTIMIZATION ============\n');

%% ---- Load ----
S = load('Accuracy_Data.mat');
X = double(S.Train_Feat);          % 125 x 13
y = double(S.Train_Label(:));      % 125 x 1
y(y~=0) = 1;
X(~isfinite(X)) = 0;
% Log transform: features are strictly positive and span ~6 orders of magnitude.
X = sign(X).*log1p(abs(X));
N = size(X,1); D = size(X,2);
posClass = 1;
fprintf('Samples=%d Features=%d  class0=%d class1=%d  (log1p-transformed)\n', N, D, sum(y==0), sum(y==1));

%% ---- Hyperparameters ----
K        = 5;       % CV folds
nSeeds   = 7;       % ensemble networks per fold (average probabilities)
epochs   = 500;
lr0      = 1e-2;
wd       = 5e-4;    % L2 weight decay (reduces train/val gap)
hidden1  = 32;
hidden2  = 16;
dropP    = 0.40;

%% ---- Stratified k-fold assignment ----
rng(42,'twister');
foldId = stratifiedKFold(y, K);

%% ---- Cross-validated predictions ----
cvScore = zeros(N,1);     % averaged P(class=1)
cvPred  = zeros(N,1);
trainAccFolds = zeros(K,1);

for f = 1:K
    teIdx = find(foldId==f);
    trIdx = find(foldId~=f);

    % standardize using training stats
    mu = mean(X(trIdx,:),1);
    sg = std(X(trIdx,:),0,1); sg(sg==0) = 1;
    Xtr = (X(trIdx,:)-mu)./sg;
    Xte = (X(teIdx,:)-mu)./sg;
    ytr = y(trIdx);  yte = y(teIdx);

    % class weights (inverse frequency)
    n0 = sum(ytr==0); n1 = sum(ytr==1);
    cw = [ (n0+n1)/(2*n0); (n0+n1)/(2*n1) ];   % [w0; w1]

    % ensemble of nSeeds nets, average probabilities
    teProbAcc = zeros(numel(teIdx),1);
    trProbAcc = zeros(numel(trIdx),1);
    for s = 1:nSeeds
        rng(1000*f+s,'twister');
        net = buildNet(D,hidden1,hidden2,dropP);
        net = trainNet(net, Xtr, ytr, cw, epochs, lr0, wd);
        teProbAcc = teProbAcc + predictProb1(net, Xte);
        trProbAcc = trProbAcc + predictProb1(net, Xtr);
    end
    teProb = teProbAcc/nSeeds;
    trProb = trProbAcc/nSeeds;

    cvScore(teIdx) = teProb;
    cvPred(teIdx)  = double(teProb>=0.5);
    trainAccFolds(f) = mean((trProb>=0.5)==ytr);
    fprintf('  fold %d/%d done (train acc %.3f)\n', f, K, trainAccFolds(f));
end

%% ---- Metrics @ 0.5 threshold ----
[P,R,F1,acc,TP,FP,TN,FN] = binMetrics(y, cvPred, posClass);
AUC = aucMW(y, cvScore, posClass);
trainAcc = mean(trainAccFolds);

%% ---- Threshold tuning to maximize F1 (reported separately) ----
ths = linspace(0.05,0.95,91); bestF1=-1; bestTh=0.5;
for t = ths
    pr = double(cvScore>=t);
    [~,~,f1t] = binMetrics(y, pr, posClass);
    if f1t>bestF1, bestF1=f1t; bestTh=t; end
end
prT = double(cvScore>=bestTh);
[Pt,Rt,F1t,accT,TPt,FPt,TNt,FNt] = binMetrics(y, prT, posClass);

%% ---- Report ----
fprintf('\n================ CROSS-VALIDATED RESULTS (threshold 0.50) ================\n');
fprintf('Precision        : %.2f%%\n', P*100);
fprintf('Recall           : %.2f%%\n', R*100);
fprintf('F1 Score         : %.2f%%\n', F1*100);
fprintf('ROC-AUC          : %.4f\n',   AUC);
fprintf('Validation Acc   : %.2f%%\n', acc*100);
fprintf('Training Acc      : %.2f%%\n', trainAcc*100);
fprintf('Overall Accuracy : %.2f%%\n', acc*100);
fprintf('Confusion [TP FP TN FN] = [%d %d %d %d]\n', TP,FP,TN,FN);

fprintf('\n---- F1-optimal threshold = %.2f ----\n', bestTh);
fprintf('Precision %.2f%% | Recall %.2f%% | F1 %.2f%% | Acc %.2f%% | [TP FP TN FN]=[%d %d %d %d]\n', ...
    Pt*100,Rt*100,F1t*100,accT*100,TPt,FPt,TNt,FNt);

%% ---- Save outputs ----
fid=fopen('metrics_summary.csv','w');
fprintf(fid,'Threshold,Precision,Recall,F1,AUC,ValAccuracy,TrainAccuracy,TP,FP,TN,FN\n');
fprintf(fid,'0.50,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n',P,R,F1,AUC,acc,trainAcc,TP,FP,TN,FN);
fprintf(fid,'%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n',bestTh,Pt,Rt,F1t,AUC,accT,trainAcc,TPt,FPt,TNt,FNt);
fclose(fid);

% Bar chart
f1fig=figure('Visible','off','Position',[100 100 720 500]);
vals=[P R F1 AUC acc trainAcc];
b=bar(vals); ylim([0 1]); grid on;
set(gca,'XTickLabel',{'Precision','Recall','F1','AUC','ValAcc','TrainAcc'});
ylabel('Score'); title('Deep-Learning Classifier — Cross-Validated Metrics');
xt=1:numel(vals);
text(xt, vals+0.02, compose('%.3f',vals),'HorizontalAlignment','center','FontSize',9);
saveas(f1fig,'best_metrics_bar.png'); close(f1fig);

% ROC curve
[fpr,tpr] = rocCurve(y, cvScore, posClass);
f2fig=figure('Visible','off','Position',[100 100 600 500]);
plot(fpr,tpr,'LineWidth',2); hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;
title(sprintf('ROC Curve (AUC = %.4f)',AUC));
saveas(f2fig,'best_roc_curve.png'); close(f2fig);

% Confusion matrix figure
f3fig=figure('Visible','off','Position',[100 100 500 450]);
Cm=[TN FP; FN TP];
imagesc(Cm); colormap(gray); colorbar; axis equal tight;
set(gca,'XTick',1:2,'XTickLabel',{'Pred 0','Pred 1'},'YTick',1:2,'YTickLabel',{'True 0','True 1'});
title('Confusion Matrix (thr 0.50)');
for i=1:2, for j=1:2
    text(j,i,num2str(Cm(i,j)),'HorizontalAlignment','center','Color','r','FontWeight','bold','FontSize',14);
end, end
saveas(f3fig,'confusion_matrix.png'); close(f3fig);

save('dl_results.mat','P','R','F1','AUC','acc','trainAcc','bestTh','cvScore','cvPred','y');
fprintf('\nSaved: metrics_summary.csv, best_metrics_bar.png, best_roc_curve.png, confusion_matrix.png, dl_results.mat\n');
fprintf('DONE_OPTIMIZE\n');
end

%% ================= helpers =================
function foldId = stratifiedKFold(y, K)
    foldId = zeros(numel(y),1);
    cls = unique(y);
    for c = cls'
        idx = find(y==c);
        idx = idx(randperm(numel(idx)));
        for i=1:numel(idx)
            foldId(idx(i)) = mod(i-1,K)+1;
        end
    end
end

function net = buildNet(D,h1,h2,dropP)
    layers = [
        featureInputLayer(D,'Normalization','none','Name','in')
        fullyConnectedLayer(h1,'Name','fc1')
        reluLayer('Name','relu1')
        dropoutLayer(dropP,'Name','drop1')
        fullyConnectedLayer(h2,'Name','fc2')
        reluLayer('Name','relu2')
        fullyConnectedLayer(2,'Name','fc3')
        softmaxLayer('Name','sm')];
    net = dlnetwork(layers);
end

function net = trainNet(net, X, y, cw, epochs, lr0, wd)
    % full-batch custom training loop, Adam + decoupled weight decay, weighted CE
    dlX = dlarray(X','CB');                 % D x N
    T = zeros(2,numel(y));                  % one-hot: row1=class0, row2=class1
    T(1, y==0)=1; T(2, y==1)=1;
    dlT = dlarray(T,'CB');
    w = dlarray(cw(:),'C');                 % 2 x 1
    avgG=[]; avgSq=[]; iter=0;
    for ep=1:epochs
        iter=iter+1;
        lr = lr0/(1+0.002*ep);              % gentle decay
        [~,grads] = dlfeval(@modelLoss, net, dlX, dlT, w);
        [net,avgG,avgSq] = adamupdate(net,grads,avgG,avgSq,iter,lr);
        % decoupled weight decay on weights (not biases)
        if wd>0
            L = net.Learnables;
            for r=1:height(L)
                if strcmp(L.Parameter{r},'Weights')
                    L.Value{r} = L.Value{r}*(1-lr*wd);
                end
            end
            net.Learnables = L;
        end
    end
end

function [loss,grads] = modelLoss(net, X, T, w)
    Y = forward(net, X);                    % 2 x N softmax
    Y = max(Y,1e-12);
    sw = sum(w.*T,1);                       % 1 x N per-sample weight
    ce = -sum(T.*log(Y),1);                 % 1 x N
    loss = sum(sw.*ce)/sum(sw);
    grads = dlgradient(loss, net.Learnables);
end

function p1 = predictProb1(net, X)
    Y = predict(net, dlarray(X','CB'));     % 2 x N
    p1 = extractdata(Y(2,:))';              % P(class=1)
end

function [P,R,F1,acc,TP,FP,TN,FN] = binMetrics(y, yhat, pos)
    neg = ~ (y==pos);
    TP = sum(yhat==pos & y==pos);
    FP = sum(yhat==pos & neg);
    TN = sum(yhat~=pos & neg);
    FN = sum(yhat~=pos & y==pos);
    P = TP/(TP+FP+eps);
    R = TP/(TP+FN+eps);
    F1 = 2*P*R/(P+R+eps);
    acc = (TP+TN)/numel(y);
end

function AUC = aucMW(y, score, pos)
    % Mann-Whitney U based AUC
    posS = score(y==pos); negS = score(y~=pos);
    nP=numel(posS); nN=numel(negS);
    if nP==0||nN==0, AUC=NaN; return; end
    [~,ord]=sort(score); ranks=zeros(size(score));
    ranks(ord)=1:numel(score);
    % average ties
    [us,~,ic]=unique(score);
    for k=1:numel(us)
        m=ic==k;
        ranks(m)=mean(ranks(m));
    end
    sumRpos=sum(ranks(y==pos));
    AUC=(sumRpos - nP*(nP+1)/2)/(nP*nN);
end

function [fpr,tpr] = rocCurve(y, score, pos)
    th=unique([0;sort(score);1]);
    th=sort(th,'descend');
    fpr=zeros(numel(th),1); tpr=zeros(numel(th),1);
    P=sum(y==pos); Ng=sum(y~=pos);
    for i=1:numel(th)
        pr=score>=th(i);
        tpr(i)=sum(pr & y==pos)/max(P,1);
        fpr(i)=sum(pr & y~=pos)/max(Ng,1);
    end
    fpr=[0;fpr;1]; tpr=[0;tpr;1];
    [fpr,o]=sort(fpr); tpr=tpr(o);
end
