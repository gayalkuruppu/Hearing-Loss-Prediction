%% Initialization

%% ================ Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('mydataset.csv');
X_train = data(2:1000, 1:12);
y_train = data(2:1000, 13);

X_test = data(1001:1391, 1:12);
y_test = data(1001:1391, 13);
m = length(y_train);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.1f \n', [X_train(1:10,:) y_train(1:10,:)]');

%fprintf('Program paused. Press enter to continue.\n');
%pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X_train mu sigma] = featureNormalize(X_train);

% Add intercept term to X
X_train = [ones(m, 1) X_train];


%% ================ Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(13, 1);
[theta, J_history] = gradientDescentMulti(X_train, y_train, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');


y_pred = zeros(391, 1);

for query = 1:rows(X_test)
    query_i = (X_test(query, :));
    y_query = (query_i .- mu) ./ sigma;
    y_query = sum([1 y_query]' .* theta);
    y_pred(query) = y_query;
endfor

RMSE = sqrt(mean((y_test - y_pred).^2));
MAPE= mean((abs(y_test-y_pred))./y_pred)*100;

fprintf(['Root Mean Square Error:\n $%f\n'], RMSE);
fprintf(['Mean Absolute Percentage Error:\n $%f%%\n'], MAPE);
