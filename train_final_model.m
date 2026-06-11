function train_final_model()
% TRAIN_FINAL_MODEL  Train the winning configuration (E8_log_4824L) on ALL
% available data and save a deployable ensemble model for the forgery script.
%
% Winning recipe: log1p features -> z-score -> dense net [48 24] ->
% dropout 0.35, weight-decay 5e-4, class-weighted loss, 800 epochs, 7-net ensemble.
%
% Saves forgery_dl_model.mat with:
%   nets       - 1xS cell of trained dlnetworks (ensemble)
%   mu, sg     - standardization parameters (computed on log-features)
%   useLog     - logical, whether log1p transform is applied
%   thr        - decision threshold on P(forged)
%   cvMetrics  - struct of cross-validated metrics (P,R,F1,AUC,acc) from the sweep
%   cvScore,cvY- cross-validated out-of-fold scores and labels (for ROC plot)

fprintf('Training final deployable model (E8_log_4824L) on all data...\n');

S = load('Accuracy_Data.mat');
X = double(S.Train_Feat); y = double(S.Train_Label(:)); y(y~=0)=1;
X(~isfinite(X))=0;
N = size(X,1); D = size(X,2); %#ok<NASGU>

% ---- winning hyperparameters ----
useLog = true;
hvec   = [48 24];
drop   = 0.35;
wd     = 5e-4;
epochs = 800;
nSeeds = 7;
thr    = 0.5;

% ---- preprocessing (fit on ALL data for deployment) ----
if useLog, Xp = sign(X).*log1p(abs(X)); else, Xp = X; end
mu = mean(Xp,1); sg = std(Xp,0,1); sg(sg==0)=1;
Xs = (Xp-mu)./sg;

% ---- class weights ----
n0 = sum(y==0); n1 = sum(y==1);
cw = [(n0+n1)/(2*n0); (n0+n1)/(2*n1)];

% ---- train ensemble on all data ----
nets = cell(1,nSeeds);
for s = 1:nSeeds
    rng(20240+s,'twister');
    net = buildNet(size(Xs,2),hvec,drop);
    net = trainNet(net,Xs,y,cw,epochs,1e-2,wd);
    nets{s} = net;
    fprintf('  trained ensemble net %d/%d\n', s, nSeeds);
end

% ---- attach honest cross-validated metrics from the sweep ----
cvMetrics = struct('P',NaN,'R',NaN,'F1',NaN,'AUC',NaN,'acc',NaN);
cvScore = []; cvY = [];
if exist('dl_results.mat','file')==2
    Rr = load('dl_results.mat');     % B (best struct), y
    if isfield(Rr,'B')
        B = Rr.B;
        cvMetrics = struct('P',B.P,'R',B.R,'F1',B.F1,'AUC',B.AUC,'acc',B.acc);
        if isfield(B,'score'), cvScore = B.score; end
    end
    if isfield(Rr,'y'), cvY = Rr.y; end
end

save('forgery_dl_model.mat','nets','mu','sg','useLog','thr','hvec','cvMetrics','cvScore','cvY');
fprintf('Saved forgery_dl_model.mat (%d-net ensemble).\n', nSeeds);
fprintf('DONE_TRAIN\n');
end

%% ---- local helpers (mirror optimize_dl2.m) ----
function net=buildNet(D,hvec,dropP)
    layers=featureInputLayer(D,'Normalization','none');
    for i=1:numel(hvec)
        layers=[layers; fullyConnectedLayer(hvec(i)); reluLayer]; %#ok<AGROW>
        if i==1, layers=[layers; dropoutLayer(dropP)]; end %#ok<AGROW>
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
