% the function for LDA
function [W, R_I] = myLDA(data)
labels = data(:,end); 
data = data(:,1:end-1);
% select the variables don't have any infomation
% to keep sw is full rank and can be inversed
R_I = find(sum(data, 1)~=0);
data = data(:, R_I);

d = size(data, 2); % the number of variables
c = size(unique(labels), 1); % the number of classes
n = size(data, 1); % the number of samples

% calculate the means of sample within different classes
m_c = zeros(c, d);
for i=1:c
    m_c(i, :) = mean(data(find(labels==i-1),:));
end
m = mean(m_c);

% calculate between-class scatter matrix
sb = zeros(d);
for i=1:c
    n_i = sum(labels==i-1);
    sb = sb + n_i*(m_c(i,:)-m)'*(m_c(i,:)-m);
end

% calculate within-class scatter matrix
sw = zeros(d);
for i=1:c
    data_c = data(find(labels==i-1),:);
    n_i = sum(labels==i-1);
    data_c = data_c-repmat(m_c(i,:),size(n_i,1),1);
    sw = sw + data_c'*data_c;
end

% calculate the eigen values and eigen vectors
% Notice that the inv(sw)*sb is non-symmetric matrix,
% The results may include complex numbers, one way to
% avoid them is adding a small diagonal matrix to sw
[V, D] = eig(inv(sw)*sb);
[D, I] = sort(diag(D), 'descend');
V = V(:, I);

W = V;
end