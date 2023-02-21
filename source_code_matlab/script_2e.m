% load training data and test data
training_data = table2array(readtable('optdigits_train.txt'));
test_data = table2array(readtable('optdigits_test.txt'));

plot_lda_2d(training_data, test_data);

function plot_lda_2d(training_data, test_data, W)
training_labels = training_data(:,end);
training_data = training_data(:,1:end-1);
test_labels = test_data(:,end);
test_data = test_data(:,1:end-1);

[W, R_I] = myLDA(training_data);
plot_data = training_data(:, R_I)*W(:,1:2);
scatter(plot_data(:,1), plot_data(:,2), 25, training_labels);hold on;
plot_data = test_data(:, R_I)*W(:,1:2); 
scatter(plot_data(:,1), plot_data(:,2), 25, test_labels);

text(plot_data(:,1), plot_data(:,2), num2str(test_labels));hold off;
end