function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Initial values
m = size(X, 1); % number of rows (examples)
num_labels = size(Theta2, 1); % 4
p = zeros(size(X, 1), 1);  % size(p)=1x1

h1 = sigmoid([ones(m, 1) X] * Theta1');  % size(h1)=(1x(2500+1)) x (2501x25) = 1x25
h2 = sigmoid([ones(m, 1) h1] * Theta2'); % size(h2)=(1x(25+1)) x (26x4) = 1x4
[value, p] = max(h2, [], 2); % p is index

end
