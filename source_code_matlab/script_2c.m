% load training data and test data
training_data = table2array(readtable('optdigits_train.txt'));
test_data = table2array(readtable('optdigits_test.txt'));

[W, comp] = myPCA(training_data);
pca_plot_2d(training_data, test_data, W);

% the function for ploting data after performing pca with keeping
% first two principal components
function pca_plot_2d(training_data, test_data, W)
training_labels = training_data(:,end);
training_data = training_data(:,1:end-1);
test_labels = test_data(:,end);
test_data = test_data(:,1:end-1);

plot_data = training_data*W(:,1:2);
scatter(plot_data(:,1), plot_data(:,2), 25, training_labels);hold on;
plot_data = test_data*W(:,1:2); 
scatter(plot_data(:,1), plot_data(:,2), 25, test_labels, 'filled');
text(plot_data(:,1), plot_data(:,2), num2str(test_labels));hold off;
end