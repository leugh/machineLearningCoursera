function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h_0 = sigmoid(X*theta);
temp1 = sum(-y.*log(h_0));
temp2 = sum((1-y).*log(1-h_0));
non_reg = (temp1 - temp2)/m;
theta(1) = 0;
reg = (lambda/(2*m))*(theta' * theta);
J = non_reg + reg;

error = X'*(h_0 - y);
non_reg = error/m;
reg = lambda/m * theta,
grad = non_reg + reg;





% =============================================================

end
