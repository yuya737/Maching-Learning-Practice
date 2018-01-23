function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

predicted = X * theta; % 12*1 matrix;
J = sum(((predicted - y) .^ 2)) / (2 * m);
J = J + (lambda * sum(theta(2:end) .^ 2) / (2 * m));

for j = 1 : size((theta), 1);
  if (j == 1)
    grad(j) = sum((predicted - y) .* X(:, j)) / m;
  else
    grad(j) = (sum((predicted - y) .* X(:, j)) + lambda * theta(j)) / m;
  end;
end;


% =========================================================================

grad = grad(:);

end
