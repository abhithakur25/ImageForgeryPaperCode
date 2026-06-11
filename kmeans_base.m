function [idx, C] = kmeans_base(X, k, maxIter)
% KMEANS_BASE  Base-MATLAB replacement for kmeans (no Statistics Toolbox).
%
%   [idx, C] = kmeans_base(X, k)
%   [idx, C] = kmeans_base(X, k, maxIter)
%
%   Lloyd's algorithm with k-means++ style seeding and 'singleton' empty-cluster
%   handling (an emptied centroid is moved to the point farthest from its
%   current centroid). Mirrors kmeans(X, k, 'EmptyAction','singleton'):
%     idx - n x 1 cluster index (1..k) for each row of X
%     C   - k x size(X,2) cluster centroids
%
%   The global RNG state is saved and restored so callers are unaffected.

if nargin < 3 || isempty(maxIter), maxIter = 100; end
[n, d] = size(X);
k = max(1, min(k, n));

rngState = rng;                 % save caller RNG, restore on exit
cleanup  = onCleanup(@() rng(rngState));
rng(0, 'twister');              % deterministic seeding

% ---- k-means++ seeding ----
C = zeros(k, d);
C(1,:) = X(randi(n), :);
D2min = sum((X - C(1,:)).^2, 2);
for c = 2:k
    probs = D2min / max(sum(D2min), eps);
    cp = cumsum(probs);
    r = rand;
    sel = find(cp >= r, 1, 'first');
    if isempty(sel), sel = randi(n); end
    C(c,:) = X(sel,:);
    D2min = min(D2min, sum((X - C(c,:)).^2, 2));
end

% ---- Lloyd iterations ----
idx = ones(n,1);
for it = 1:maxIter
    % assignment step
    D2 = sum(X.^2,2) + sum(C.^2,2)' - 2*(X*C');   % n x k
    [~, newIdx] = min(D2, [], 2);
    if it > 1 && isequal(newIdx, idx)
        idx = newIdx; break;
    end
    idx = newIdx;
    % update step
    for c = 1:k
        members = (idx == c);
        if any(members)
            C(c,:) = mean(X(members,:), 1);
        else
            % EmptyAction 'singleton': move centroid to farthest point
            Dall = sum((X - C(idx,:)).^2, 2);
            [~, far] = max(Dall);
            C(c,:) = X(far,:);
            idx(far) = c;
        end
    end
end
end
