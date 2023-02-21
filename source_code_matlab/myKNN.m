% the function for knn
function test_err = myKNN(training_data, test_data, k)
training_labels = training_data(:,end);
training_data = training_data(:,1:end-1);
test_labels = test_data(:,end);
test_data = test_data(:,1:end-1);

% calculate the distance between any pairs between training data 
% and test data, and return the nearest k samples
[D,I] = pdist2(training_data,test_data,'euclidean','Smallest',k);
n_test=size(test_data, 1);
pred_labels=zeros(n_test, 1);
% calculate the labels for test data using the neighbors in the 
% training data
for i=1:n_test
    counts=tabulate(training_labels(I(:,i),1));
    [v_m, i_m] = max(counts(:,2));
    pred_labels(i)=counts(i_m,1);
end
test_err=size(find((pred_labels-test_labels)~=0), 1)/n_test;
end