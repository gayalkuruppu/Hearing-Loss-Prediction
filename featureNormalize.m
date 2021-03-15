function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 13));
sigma = zeros(1, size(X, 13));

mu = mean(X_norm);
sigma = std(X_norm);
X_norm = (X_norm .- mu) ./ sigma;

end
