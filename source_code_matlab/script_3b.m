training_data = table2array(readtable('face_train_data_960.txt'));
test_data = table2array(readtable('face_test_data_960.txt'));
faces_data = [training_data;test_data];

[W, comp] = myPCA(training_data);
[test_errs, min_err, best_k] = find_opt_k(training_data, test_data, 1:2:7, W, comp);

% the same function in PCA.m and KNN.m
function [test_errs, min_err, best_k] = find_opt_k(training_data, test_data, ks, W, comp)
training_data = [training_data(:,1:end-1)*W(:,1:comp) training_data(:,end)];
test_data = [test_data(:,1:end-1)*W(:,1:comp) test_data(:,end)];
test_errs = zeros(length(ks), 1);
for i=1:length(ks)
    test_errs(i) = myKNN(training_data, test_data, ks(i));
end
[v_m, i_m] = min(test_errs);
min_err = v_m;
best_k = ks(i_m);
end