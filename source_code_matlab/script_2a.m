% load training data and test data
training_data = table2array(readtable('optdigits_train.txt'));
test_data = table2array(readtable('optdigits_test.txt'));

[test_errs, min_err, best_k] = find_opt_k(training_data, test_data, 1:2:7);

% the function for finding the optimal parameter k in knn
% and calculating the test errors over a range of k values
function [test_errs, min_err, best_k] = find_opt_k(training_data, test_data, ks)
test_errs = zeros(length(ks), 1);
for i=1:length(ks)
    test_errs(i) = myKNN(training_data, test_data, ks(i));
end
[v_m, i_m] = min(test_errs);
min_err = v_m;
best_k = ks(i_m);
end