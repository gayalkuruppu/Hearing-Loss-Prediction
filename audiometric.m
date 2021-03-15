%% Initialization

%% ================ Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = csvread('mydataset.csv');
X = data(2:1000, 1:12);
y = data(2:1000, 13);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.1f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Choose some alpha value
alpha = 0.3;
num_iters = 100;

% Init Theta and Run Gradient Descent 
theta = zeros(13, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the Hearing Loss for input [1,44,0,0,15,20,15,20,20,15,15,10]
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
predict = [1,69,1,1,10,10,5,10,45,50,35,30]; % give inputs here to predict
predict = (predict .- mu) ./ sigma;
loss = sum([1 predict]' .* theta); 

fprintf(['Predicted Mesured hearing loss(dB)_Y' ...
         '(using gradient descent):\n $%f\n'], loss);
