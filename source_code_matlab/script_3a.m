training_data = table2array(readtable('face_train_data_960.txt'));
test_data = table2array(readtable('face_test_data_960.txt'));
faces_data = [training_data;test_data];
[W, comp] = myPCA(faces_data);
plot_eigenfaces(W, 5);

% the function for plotting eigenfaces
%plot_eigenfaces(W, 5);
function plot_eigenfaces(W, k)
for i=1:k
    subplot(ceil(k/2), 2, i)
    imagesc(reshape(W(:,i),32,30)')
end 
end