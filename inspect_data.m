% Inspect Accuracy_Data.mat contents
S = load('Accuracy_Data.mat');
fn = fieldnames(S);
fprintf('Variables in Accuracy_Data.mat:\n');
for i=1:numel(fn)
    v = S.(fn{i});
    fprintf('  %s : class=%s, size=[%s]\n', fn{i}, class(v), num2str(size(v)));
end

if isfield(S,'Train_Feat')
    F = double(S.Train_Feat);
    fprintf('\nTrain_Feat: rows=%d cols=%d\n', size(F,1), size(F,2));
    fprintf('  min=%.4g max=%.4g  anyNaN=%d anyInf=%d\n', min(F(:)), max(F(:)), any(isnan(F(:))), any(isinf(F(:))));
end
if isfield(S,'Train_Label')
    L = S.Train_Label(:);
    fprintf('\nTrain_Label: n=%d\n', numel(L));
    u = unique(L);
    fprintf('  unique labels: %s\n', num2str(u'));
    for k=1:numel(u)
        fprintf('    label %g : count %d\n', u(k), sum(L==u(k)));
    end
end
fprintf('\nDONE_INSPECT\n');
