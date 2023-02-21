% load training data and test data pair 1
training_data = table2array(readtable('training_data1.txt'));
test_data = table2array(readtable('test_data1.txt'));

results = multiGaussian(training_data, test_data, 1);
disp('Parameters for model 1 on dataset1: pc1, pc2, mu1, mu2, m1_s1, m1_s2, m1_err');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 2);
disp('Parameters for model 2 on dataset1: pc1, pc2, mu1, mu2, m2_s, m2_err');
disp('');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 3);
disp('Parameters for model 3 on dataset1: pc1, pc2, mu1, mu2, m3_s1, m3_s2, m3_err');
disp('');
%celldisp(results);
disp(results);

% load training data and test data pair 2
training_data = table2array(readtable('training_data2.txt'));
test_data = table2array(readtable('test_data2.txt'));

results = multiGaussian(training_data, test_data, 1);
disp('Parameters for model 1 on dataset2: pc1, pc2, mu1, mu2, m1_s1, m1_s2, m1_err');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 2);
disp('Parameters for model 2 on dataset2: pc1, pc2, mu1, mu2, m2_s, m2_err');
disp('');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 3);
disp('Parameters for model 3 on dataset2: pc1, pc2, mu1, mu2, m3_s1, m3_s2, m3_err');
disp('');
%celldisp(results);
disp(results);

% load training data and test data pair 3
training_data = table2array(readtable('training_data3.txt'));
test_data = table2array(readtable('test_data3.txt'));

results = multiGaussian(training_data, test_data, 1);
disp('Parameters for model 1 on dataset3: pc1, pc2, mu1, mu2, m1_s1, m1_s2, m1_err');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 2);
disp('Parameters for model 2 on dataset3: pc1, pc2, mu1, mu2, m2_s, m2_err');
disp('');
%celldisp(results);
disp(results);

results = multiGaussian(training_data, test_data, 3);
disp('Parameters for model 3 on dataset3: pc1, pc2, mu1, mu2, m3_s1, m3_s2, m3_err');
disp('');
%celldisp(results);
disp(results);

% Build multi_gaussian models
function results = multiGaussian(training_data, test_data, model_no)
training_labels = training_data(:,end);
training_data = training_data(:,1:end-1);
test_labels = test_data(:,end);
test_data = test_data(:,1:end-1);

n = size(training_data, 1); % the number of samples
d = size(training_data, 2); % the nunber of dimensions
n1 = size(find(training_labels==1), 1); % the number of samples within class 1
n2 = size(find(training_labels==2), 1); % the number of samples within class 1
indice1 = find(training_labels==1); % the indice of class 1 samples in training data
indice2 = find(training_labels==2); % the indice of class 2 samples in training data

pc1 = n1/n; % calculate the prior prob for class 1
pc2 = n2/n; % calculate the prior prob for class 2

% calculate the mean \mu_{1} of samples labelled class 1
mu1 = sum(training_data(find(training_labels==1),:), 1)./n1;
% calculate the mean \mu_{2} of samples labelled class 2
mu2 = sum(training_data(find(training_labels==2),:), 1)./n2;

% calculate S_{1} for model 1
m1_s1 = zeros(d, d);
for i = 1:n1
    m1_s1 = m1_s1 + (training_data(indice1(i),:)-mu1).'*(training_data(indice1(i),:)-mu1);
end
m1_s1 = m1_s1./n1;

% calculate S_{2} for model 1
m1_s2 = zeros(d, d);
for i = 1:n2
    m1_s2 = m1_s2 + (training_data(indice2(i),:)-mu2).'*(training_data(indice2(i),:)-mu2);
end
m1_s2 = m1_s2./n2;

% Calculate S for model 2
m2_s = pc1.*m1_s1 + pc2.*m1_s2;

% calculate \alpha_{1} for model 3
m3_s1 = zeros(d);
m3_s1 = var(training_data(indice1,:)-mu1, 1, 1);
m3_s1 = diag(m3_s1);

% calculate \alpha_{2} for model 3
m3_s2 = zeros(d);
m3_s2 = var(training_data(indice2,:)-mu2, 1, 1);
m3_s2 = diag(m3_s2);

% calculate test error for model 1
n_ = size(test_data, 1);
pred_labels = zeros(n_, 1);
for i=1:n_
    x = test_data(i, :);
    m1_c1 = -d/2 * log(2*pi)-1/2*log(det(m1_s1))-1/2*(x-mu1)*inv(m1_s1)*(x-mu1).'+log(pc1);
    m1_c2 = -d/2 * log(2*pi)-1/2*log(det(m1_s2))-1/2*(x-mu2)*inv(m1_s2)*(x-mu2).'+log(pc2);
    if m1_c1 - m1_c2 > 0
        pred_labels(i) = 1;
    else
        pred_labels(i) = 2;
    end
end
m1_err = sum(abs(test_labels - pred_labels), 1)/n_;

% % calculate test error for model 2
for i=1:n_
    x = test_data(i, :);
    m1_c1 = -d/2 * log(2*pi)-1/2*log(det(m2_s))-1/2*(x-mu1)*inv(m2_s)*(x-mu1).'+log(pc1);
    m1_c2 = -d/2 * log(2*pi)-1/2*log(det(m2_s))-1/2*(x-mu2)*inv(m2_s)*(x-mu2).'+log(pc2);
    pred_labels(i) = m1_c1 - m1_c2;
    if m1_c1 - m1_c2 > 0
        pred_labels(i) = 1;
    else
        pred_labels(i) = 2;
    end
end
m2_err = sum(abs(test_labels - pred_labels), 1)/n_;

% calculate test error for model 3
for i=1:n_
    x = test_data(i, :);
    m1_c1 = -d/2 * log(2*pi)-1/2*log(det(m3_s1))-1/2*(x-mu1)*inv(m3_s1)*(x-mu1).'+log(pc1);
    m1_c2 = -d/2 * log(2*pi)-1/2*log(det(m3_s2))-1/2*(x-mu2)*inv(m3_s2)*(x-mu2).'+log(pc2);
    pred_labels(i) = m1_c1 - m1_c2;
    if m1_c1 - m1_c2 > 0
        pred_labels(i) = 1;
    else
        pred_labels(i) = 2;
    end
end
m3_err = sum(abs(test_labels - pred_labels), 1)/n_;

% switch model_no
%     case 1
%         results = {pc1, pc2, mu1, mu2, m1_s1, m1_s2, m1_err};
%     case 2
%         results = {pc1, pc2, mu1, mu2, m2_s, m2_err};
%     case 3
%         results = {pc1, pc2, mu1, mu2, m3_s1, m3_s2, m3_err};
% end

switch model_no
    case 1
        results = m1_err;
    case 2
        results = m2_err;
    case 3
        results = m3_err;
end

end
