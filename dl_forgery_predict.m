function [label, pForged] = dl_forgery_predict(featRows, modelFile)
% DL_FORGERY_PREDICT  Classify feature rows with the optimized DL forgery model.
%
%   [label, pForged] = dl_forgery_predict(featRows)
%   [label, pForged] = dl_forgery_predict(featRows, 'forgery_dl_model.mat')
%
%   featRows : M x 13 matrix of features (same layout as Train_Feat)
%   label    : M x 1 predicted class (1 = forged, 0 = authentic)
%   pForged  : M x 1 ensemble probability of the forged class
%
% Loads the ensemble + preprocessing saved by train_final_model.m and applies
% the identical log/standardize pipeline used during training.

if nargin < 2 || isempty(modelFile), modelFile = 'forgery_dl_model.mat'; end
if exist(modelFile,'file') ~= 2
    error('Model file "%s" not found. Run train_final_model.m first.', modelFile);
end
M = load(modelFile);

X = double(featRows); X(~isfinite(X)) = 0;
if M.useLog, X = sign(X).*log1p(abs(X)); end
Xs = (X - M.mu)./M.sg;

% ensemble-average the forged-class probability
acc = zeros(size(Xs,1),1);
for s = 1:numel(M.nets)
    Y = predict(M.nets{s}, dlarray(Xs','CB'));   % 2 x M softmax
    acc = acc + extractdata(Y(2,:))';
end
pForged = acc / numel(M.nets);
label   = double(pForged >= M.thr);
end
