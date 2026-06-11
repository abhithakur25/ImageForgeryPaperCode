function optimize_dl2()
% OPTIMIZE_DL2  Repeated stratified CV + small config sweep to maximize all
% metrics for tree/forgery detection. Base MATLAB + Deep Learning Toolbox only.

clc;
fprintf('========= TREE/FORGERY DETECTION : REPEATED-CV OPTIMIZATION (PARALLEL) =========\n');

% ---- start parallel pool (process-based) ----
tStart=tic;
p=gcp('nocreate');
if isempty(p)
    c=parcluster('Processes');
    nC=min(feature('numcores'), c.NumWorkers);   % respect cluster worker cap
    p=parpool(c, nC);
end
fprintf('Parallel pool: %d workers\n', p.NumWorkers);

S = load('Accuracy_Data.mat');
Xraw = double(S.Train_Feat);
y = double(S.Train_Label(:)); y(y~=0)=1;
Xraw(~isfinite(Xraw))=0;
N=size(Xraw,1); D=size(Xraw,2); pos=1;
fprintf('Samples=%d Features=%d class0=%d class1=%d\n',N,D,sum(y==0),sum(y==1));

K=5; R=8; nSeeds=7;            % folds, repeats, seed-ensemble per fold

% ---- configuration sweep ----
% columns: name, useLog, dropout, weightDecay, hiddenLayerVector, epochs
cfgs = {
 'C3_log_4824',  true,  0.35, 5e-4, [48 24],     500;   % prior winner (baseline)
 'C1_log_3216',  true,  0.30, 1e-4, [32 16],     400;   % prior co-leader
 'E1_log_6432',  true,  0.30, 5e-4, [64 32],     500;
 'E2_log_644816',true,  0.30, 5e-4, [64 48 16],  500;   % deeper (3 hidden)
 'E3_log_321608',true,  0.25, 1e-4, [32 16 8],   500;   % deeper, slim
 'E4_log_4824',  true,  0.40, 1e-3, [48 24],     600;   % stronger reg
 'E5_log_9648',  true,  0.35, 1e-3, [96 48],     500;   % wider
 'E6_raw_4824',  false, 0.30, 5e-4, [48 24],     500;   % raw features (AUC)
 'E7_raw_644816',false, 0.30, 5e-4, [64 48 16],  500;   % raw, deeper
 'E8_log_4824L', true,  0.35, 5e-4, [48 24],     800;   % longer training
};

bestScore=-inf; best=struct();
for ci=1:size(cfgs,1)
    name=cfgs{ci,1}; useLog=cfgs{ci,2}; drop=cfgs{ci,3}; wd=cfgs{ci,4};
    hvec=cfgs{ci,5}; epochs=cfgs{ci,6};

    X=Xraw; if useLog, X=sign(X).*log1p(abs(X)); end

    % Precompute deterministic fold partitions per repeat (outside parfor)
    foldIds=zeros(N,R);
    for rep=1:R
        rng(100*rep+ci,'twister');
        foldIds(:,rep)=stratifiedKFold(y,K);
    end

    % Flatten (rep,fold,seed) into independent jobs -> parallelize over workers
    [JR,JF,JS]=ndgrid(1:R,1:K,1:nSeeds);
    jobs=[JR(:) JF(:) JS(:)];
    nJobs=size(jobs,1);
    teP=cell(nJobs,1); teI=cell(nJobs,1); trA=zeros(nJobs,1);

    parfor jid=1:nJobs
        rep=jobs(jid,1); f=jobs(jid,2); s=jobs(jid,3);
        foldId=foldIds(:,rep);
        te=find(foldId==f); tr=find(foldId~=f);
        mu=mean(X(tr,:),1); sg=std(X(tr,:),0,1); sg(sg==0)=1;
        Xtr=(X(tr,:)-mu)./sg; Xte=(X(te,:)-mu)./sg;
        ytr=y(tr);
        n0=sum(ytr==0); n1=sum(ytr==1);
        cw=[(n0+n1)/(2*n0);(n0+n1)/(2*n1)];
        rng(7000*rep+100*f+s,'twister');
        net=buildNet(D,hvec,drop);
        net=trainNet(net,Xtr,ytr,cw,epochs,1e-2,wd);
        teP{jid}=predictProb1(net,Xte); teI{jid}=te;
        trA(jid)=mean((predictProb1(net,Xtr)>=0.5)==ytr);
    end

    % Reduce: average over the nSeeds nets per (rep,fold), then over repeats
    oofProb=zeros(N,1); oofCnt=zeros(N,1);
    for jid=1:nJobs
        idx=teI{jid};
        oofProb(idx)=oofProb(idx)+teP{jid}/nSeeds;
        oofCnt(idx)=oofCnt(idx)+1/nSeeds;
    end
    score=oofProb./max(oofCnt,1);
    pred=double(score>=0.5);
    [P,Rc,F1,acc,TP,FP,TN,FN]=binMetrics(y,pred,pos);
    AUC=aucMW(y,score,pos); trainAcc=mean(trA);
    combo=F1+AUC+acc;
    fprintf('%-12s | P=%.3f R=%.3f F1=%.3f AUC=%.4f valAcc=%.3f trAcc=%.3f | combo=%.3f\n',...
        name,P,Rc,F1,AUC,acc,trainAcc,combo);
    if combo>bestScore
        bestScore=combo;
        best=struct('name',name,'score',score,'pred',pred,'P',P,'R',Rc,'F1',F1,...
            'AUC',AUC,'acc',acc,'trainAcc',trainAcc,'TP',TP,'FP',FP,'TN',TN,'FN',FN,...
            'useLog',useLog,'drop',drop,'wd',wd,'hvec',hvec,'epochs',epochs);
    end
end

B=best; score=B.score;
% threshold tuning
ths=linspace(0.05,0.95,181); bF1=-1; bTh=0.5;
for t=ths
    [~,~,f1t]=binMetrics(y,double(score>=t),pos);
    if f1t>bF1, bF1=f1t; bTh=t; end
end
[Pt,Rt,F1t,accT,TPt,FPt,TNt,FNt]=binMetrics(y,double(score>=bTh),pos);

fprintf('\n================ BEST CONFIG: %s ================\n',B.name);
fprintf('Precision        : %.2f%%\n',B.P*100);
fprintf('Recall           : %.2f%%\n',B.R*100);
fprintf('F1 Score         : %.2f%%\n',B.F1*100);
fprintf('ROC-AUC          : %.4f\n',B.AUC);
fprintf('Validation Acc   : %.2f%%\n',B.acc*100);
fprintf('Training Acc      : %.2f%%\n',B.trainAcc*100);
fprintf('Overall Accuracy : %.2f%%\n',B.acc*100);
fprintf('Confusion [TP FP TN FN]=[%d %d %d %d]\n',B.TP,B.FP,B.TN,B.FN);
fprintf('---- F1-optimal threshold=%.3f: P=%.2f%% R=%.2f%% F1=%.2f%% Acc=%.2f%% [%d %d %d %d]\n',...
    bTh,Pt*100,Rt*100,F1t*100,accT*100,TPt,FPt,TNt,FNt);

% save artifacts
fid=fopen('metrics_summary.csv','w');
fprintf(fid,'Config,Threshold,Precision,Recall,F1,AUC,ValAccuracy,TrainAccuracy,TP,FP,TN,FN\n');
fprintf(fid,'%s,0.50,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n',B.name,B.P,B.R,B.F1,B.AUC,B.acc,B.trainAcc,B.TP,B.FP,B.TN,B.FN);
fprintf(fid,'%s,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d\n',B.name,bTh,Pt,Rt,F1t,B.AUC,accT,B.trainAcc,TPt,FPt,TNt,FNt);
fclose(fid);

f1fig=figure('Visible','off','Position',[100 100 740 500]);
vals=[B.P B.R B.F1 B.AUC B.acc B.trainAcc];
bar(vals); ylim([0 1.05]); grid on;
set(gca,'XTickLabel',{'Precision','Recall','F1','AUC','ValAcc','TrainAcc'});
ylabel('Score'); title(sprintf('Best (%s) — Repeated-CV Metrics',B.name));
text(1:numel(vals),vals+0.02,compose('%.3f',vals),'HorizontalAlignment','center','FontSize',9);
saveas(f1fig,'best_metrics_bar.png'); close(f1fig);

[fpr,tpr]=rocCurve(y,score,pos);
f2fig=figure('Visible','off','Position',[100 100 600 500]);
plot(fpr,tpr,'LineWidth',2); hold on; plot([0 1],[0 1],'k--');
xlabel('False Positive Rate'); ylabel('True Positive Rate'); grid on;
title(sprintf('ROC Curve (AUC=%.4f)',B.AUC)); saveas(f2fig,'best_roc_curve.png'); close(f2fig);

f3fig=figure('Visible','off','Position',[100 100 500 450]);
Cm=[B.TN B.FP; B.FN B.TP]; imagesc(Cm); colormap(gray); colorbar; axis equal tight;
set(gca,'XTick',1:2,'XTickLabel',{'Pred 0','Pred 1'},'YTick',1:2,'YTickLabel',{'True 0','True 1'});
title('Confusion Matrix (thr 0.50)');
for i=1:2,for j=1:2,text(j,i,num2str(Cm(i,j)),'HorizontalAlignment','center','Color','r','FontWeight','bold','FontSize',14);end,end
saveas(f3fig,'confusion_matrix.png'); close(f3fig);

save('dl_results.mat','B','bTh','y');
fprintf('\nTotal wall time: %.1f min\n', toc(tStart)/60);
fprintf('Saved artifacts. DONE_OPTIMIZE\n');
end

%% helpers
function foldId=stratifiedKFold(y,K)
    foldId=zeros(numel(y),1);
    for c=unique(y)'
        idx=find(y==c); idx=idx(randperm(numel(idx)));
        for i=1:numel(idx), foldId(idx(i))=mod(i-1,K)+1; end
    end
end
function net=buildNet(D,hvec,dropP)
    % hvec: vector of hidden-layer widths (1, 2 or more layers)
    layers=featureInputLayer(D,'Normalization','none');
    for i=1:numel(hvec)
        layers=[layers; fullyConnectedLayer(hvec(i)); reluLayer]; %#ok<AGROW>
        if i==1
            layers=[layers; dropoutLayer(dropP)]; %#ok<AGROW>
        end
    end
    layers=[layers; fullyConnectedLayer(2); softmaxLayer];
    net=dlnetwork(layers);
end
function net=trainNet(net,X,y,cw,epochs,lr0,wd)
    dlX=dlarray(X','CB');
    T=zeros(2,numel(y)); T(1,y==0)=1; T(2,y==1)=1; dlT=dlarray(T,'CB');
    w=dlarray(cw(:),'C'); avgG=[]; avgSq=[];
    for ep=1:epochs
        lr=lr0/(1+0.002*ep);
        [~,grads]=dlfeval(@modelLoss,net,dlX,dlT,w);
        [net,avgG,avgSq]=adamupdate(net,grads,avgG,avgSq,ep,lr);
        if wd>0
            L=net.Learnables;
            for r=1:height(L)
                if strcmp(L.Parameter{r},'Weights'), L.Value{r}=L.Value{r}*(1-lr*wd); end
            end
            net.Learnables=L;
        end
    end
end
function [loss,grads]=modelLoss(net,X,T,w)
    Y=forward(net,X); Y=max(Y,1e-12);
    sw=sum(w.*T,1); ce=-sum(T.*log(Y),1);
    loss=sum(sw.*ce)/sum(sw); grads=dlgradient(loss,net.Learnables);
end
function p1=predictProb1(net,X)
    Y=predict(net,dlarray(X','CB')); p1=extractdata(Y(2,:))';
end
function [P,R,F1,acc,TP,FP,TN,FN]=binMetrics(y,yhat,pos)
    neg=~(y==pos);
    TP=sum(yhat==pos&y==pos); FP=sum(yhat==pos&neg);
    TN=sum(yhat~=pos&neg);    FN=sum(yhat~=pos&y==pos);
    P=TP/(TP+FP+eps); R=TP/(TP+FN+eps); F1=2*P*R/(P+R+eps); acc=(TP+TN)/numel(y);
end
function AUC=aucMW(y,score,pos)
    nP=sum(y==pos); nN=sum(y~=pos);
    if nP==0||nN==0, AUC=NaN; return; end
    [~,ord]=sort(score); ranks=zeros(size(score)); ranks(ord)=1:numel(score);
    [us,~,ic]=unique(score);
    for k=1:numel(us), m=ic==k; ranks(m)=mean(ranks(m)); end
    AUC=(sum(ranks(y==pos))-nP*(nP+1)/2)/(nP*nN);
end
function [fpr,tpr]=rocCurve(y,score,pos)
    th=sort(unique([0;score;1]),'descend');
    P=sum(y==pos); Ng=sum(y~=pos); fpr=zeros(numel(th),1); tpr=fpr;
    for i=1:numel(th)
        pr=score>=th(i);
        tpr(i)=sum(pr&y==pos)/max(P,1); fpr(i)=sum(pr&y~=pos)/max(Ng,1);
    end
    fpr=[0;fpr;1]; tpr=[0;tpr;1]; [fpr,o]=sort(fpr); tpr=tpr(o);
end
