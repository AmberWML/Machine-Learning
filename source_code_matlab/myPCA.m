% the function for PCA
function [W, comp] = myPCA(data)

labels = data(:,end);
data = data(:,1:end-1);

d = size(data, 2); % the number of variables
cov_mat = cov(data); % calculate the covariance matrix
[V, D] = eig(cov_mat); % calculate the eigen values and vectors for covariance matrix
[D, I] = sort(diag(D), 'descend'); % sort the eigen values in descend order
V = V(:, I); % rearrange eigen vectors according to eigen values

% calculate the proportion of variance explained 
prop = cumsum(D/sum(D));
comp = find(prop>0.9);
comp = comp(1);
W = V;
plot(1:d, prop,...
    'Marker','*');
xlabel('Eigenvectors');
ylabel('Prop. of var.');
end