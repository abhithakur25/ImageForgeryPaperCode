function [idx, dist] = knnsearch_base(X, Y, K)
% KNNSEARCH_BASE  Base-MATLAB replacement for knnsearch (no Statistics Toolbox).
%
%   [idx, dist] = knnsearch_base(X, Y, K)
%
%   For each query row in Y, finds the K nearest rows in X using Euclidean
%   distance. Mirrors knnsearch(X, Y, 'K', K, 'distance', 'euclidean'):
%     idx  - size(Y,1) x K indices into the rows of X (nearest first)
%     dist - size(Y,1) x K Euclidean distances (ascending per row)

if nargin < 3 || isempty(K), K = 1; end
nx = size(X,1);
K  = min(K, nx);

% squared Euclidean distances via ||y-x||^2 = ||y||^2 + ||x||^2 - 2 y*x'
Xsq = sum(X.^2, 2)';            % 1 x nx
Ysq = sum(Y.^2, 2);            % ny x 1
D2  = Ysq + Xsq - 2*(Y*X');    % ny x nx
D2(D2 < 0) = 0;                % clamp tiny negatives from round-off

[sd, si] = sort(D2, 2, 'ascend');
idx  = si(:, 1:K);
dist = sqrt(sd(:, 1:K));
end
