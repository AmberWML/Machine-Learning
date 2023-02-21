% implements Bayes Testing, return the test error

function test_error = Bayes_Testing(test_data, p1, p2, pc1, pc2)

[test_row_size, column_size] = size(test_data); % dimension of test data
errors_test = 0; % total error to be count

% classify the test set using the learned parameters p1, p2, pc1, pc2
for k = 1:test_row_size
    temp_p1 = 1; % \prod_{j=1}^{D} p_{1j}^{1-x_j}(1-p_{1j})^{x_j}
    temp_p2 = 1; % \prod_{j=1}^{D} p_{2j}^{1-x_j}(1-p_{2j})^{x_j}
    for j = 1: column_size-1
        temp_p1 = temp_p1 * p1(j)^(1-test_data(k,j)) * (1-p1(j))^test_data(k,j);
        temp_p2 = temp_p2 * p2(j)^(1-test_data(k,j)) * (1-p2(j))^test_data(k,j);
    end
    g_x = pc1*temp_p1 - pc2*temp_p2; % compute g(x)
    if (g_x>=0 && test_data(k,column_size)==2) || (g_x<0 && test_data(k,column_size)==1)
        errors_test = errors_test + 1; % update error counts if prediction and true label are different
    end
end

% get test error and print to terminal
test_error = errors_test/test_row_size; % Result of error rate using the best priors to classify test data
fprintf('Error rate on the test dataset is: \n\n');
disp(test_error);

end