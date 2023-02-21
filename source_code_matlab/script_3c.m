% load face images
training_data = table2array(readtable('face_train_data_960.txt'));
test_data = table2array(readtable('face_test_data_960.txt'));
faces_data = [training_data;test_data];

[W, comp] = myPCA(training_data);
plot_recon_data(training_data, W, [10, 50, 100], 5);

% the function for plotting reconstructed images
function plot_recon_data(raw_data, W, comps, k)
for i=1:length(comps)
    figure(i);
    recon_data = recons(raw_data, W, comps(i));
    for j=1:k
        subplot(ceil(k/2), 2, j);
        imagesc(reshape(recon_data(j,:),32,30)');
    end 
end
end

% the function for image reconstruction
% \hat{X} = XWW^{T} + \mu
function Xhat = recons(X, W, comp)
X = X(:, 1:end-1);
mu = mean(X);
Xhat = X * W(:,1:comp) * W(:,1:comp)';
Xhat = bsxfun(@plus, Xhat, mu);
end