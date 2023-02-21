% implements Bayes_Learning, returns the outputs (p1: learned Bernoulli
% parameters of the first class, p2: learned Bernoulli parameters of the
% second class, pc1: best prior of the first class, pc2: best prior of the
% second class

function [p1,p2,pc1,pc2] = Bayes_Learning(training_data, validation_data)

[train_row_size, column_size] = size (training_data); % dimension of training data
[valid_row_size, ~] = size (validation_data); % dimension of validation data
X = training_data(1:train_row_size, 1:column_size-1); %Training data

% find label counts of class 1 and class 2
R = zeros(2, train_row_size); % Label at R_{ij} is indicator if x^j in class C_i
for j = 1:train_row_size
    if training_data(j, column_size)==1
        R(1,j)=1; % x^j in class 1
    else
        R(2,j)=1; % x^j in class 2
    end
end
Count = repmat([sum(R(1,:));sum(R(2,:))],1,column_size-1); %First row is class 1 count, second row is class 2 count

% get MLE p1, p2
P = ones(2, column_size-1)-(R*X)./Count; % formula: P_{2*D}=1_{2*D}-(R_{2*n}X_{n*D})./Count
p1 = P(1,:);
p2 = P(2,:);

% Use different P(C_1) and P(C_2) on validation set
% We compute g(x)= based on priors P(C_1), P(C_2), MLE estimator p1, p2, and x_{1*D}
error_table = zeros(11,4); % build an error table with 4 columns of : sigma, P(C1), P(C2), error_rate
index = 1; % row index of error table
for sigma = [-5,-4,-3,-2,-1,0,1,2,3,4,5]
    P_C1 = 1/(1+(exp(-sigma))); % set priors using formula P(C1)=1/(1+(exp(-sigma)))
    P_C2 = 1 - P_C1;
    error_count = 0; % total number of errors to be count
    for k = 1:valid_row_size
        temp_p1 = 1; % \prod_{j=1}^{D} p_{1j}^{1-x_j}(1-p_{1j})^{x_j}
        temp_p2 = 1; % \prod_{j=1}^{D} p_{2j}^{1-x_j}(1-p_{2j})^{x_j}
        for j = 1: column_size-1
            temp_p1 = temp_p1 * p1(j)^(1-validation_data(k,j)) * (1-p1(j))^validation_data(k,j);
            temp_p2 = temp_p2 * p2(j)^(1-validation_data(k,j)) * (1-p2(j))^validation_data(k,j);
        end
        g_x = P_C1*temp_p1 - P_C2*temp_p2; % compute g(x)
        if (g_x>=0 && validation_data(k,column_size)==2) || (g_x<0 && validation_data(k,column_size)==1)
            error_count = error_count + 1; % update error counts if prediction and true label is different
        end
    end
    error_table(index,1) = sigma;
    error_table(index,2) = P_C1;
    error_table(index,3) = P_C2;
    error_table(index,4) = error_count/valid_row_size; % update error table
    index = index + 1;
end

% get the best priors
[~, I] = min(error_table(:,4)); % find row index of the lowest error rate on validation set
pc1 = error_table(I,2);
pc2 = error_table(I,3); % best priors

% print error table to terminal
fprintf('\n Error rates of all priors on validation set: \n\n');
fprintf('    sigma     P(C1)     P(C2)     error rate on validation set\n\n');
disp(error_table);

end
