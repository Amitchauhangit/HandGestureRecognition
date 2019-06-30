function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function

% Initial values
g = zeros(size(z));

% gradient of the sigmoid
g = sigmoid(z) .* (1 - sigmoid(z));

end
