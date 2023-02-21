% load training data and test data
training_data = table2array(readtable('optdigits_train.txt'));
test_data = table2array(readtable('optdigits_test.txt'));

test_errs = comb_knn_lda(training_data, test_data, [2 4 9], [1 3 5 7]);

function test_errs = comb_knn_lda(training_data, test_data, ls, ks)
test_errs = zeros(length(ls), length(ks));
[W, R_I] = myLDA(training_data);
for i=1:length(ls)
    training_data_p = [training_data(:, R_I)*W(:,1:ls(i)) training_data(:, end)];
    test_data_p = [test_data(:, R_I)*W(:,1:ls(i)) test_data(:, end)];
    for j=1:length(ks)
        test_errs(i, j)=myKNN(training_data_p, test_data_p, ks(j));
    end
end
end