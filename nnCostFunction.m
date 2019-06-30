function [J grad] = nnCostFunction(nn_params, input_layer_size, ...
                                 hidden_layer_size,num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, 
%   X, y, lambda) computes the cost and gradient of the neural network.  
% 
%   The returned parameter grad is partial derivatives of nnparameter the neural network.
%



% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1)); %25x(2500+1) = 25x2501

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); %4x(25+1)= 4x26

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Feedforward  method

summary = 0;
for j = 1:m

    y_label = y(j,:);

    a_1 = X(j,:); % one input image data size(a_1)=(1, 2500)
    a_1 = [1 a_1]; % adding 1 as bais node , therefore size(a_1)= (1,2501)
    z_1 = a_1 * Theta1';% size(z_1)= (1x2501) x (25x2501)' = (1 x 25)
    a_2 = sigmoid(z_1); % applying sigmoid function to values in z_1
    
    m_temp = size(a_2, 1); % m_temp=1
    a_2 = [ones(m_temp, 1) a_two]; % adding 1 as bias node, therefore size(a_2)=(1,26)
    z_2 = a_2 * Theta2'; % size(z_2) = (1x26) x (4x26)' = (1x4)
    a_3 = sigmoid(z_2);% applying sigmoid function to values in z_2
    h = a_3;


    sum_temp = 0;
    sum_temp = (-y_label .* log(h)) - ((1 - y_label) .* log(1 - h)); %calculating cost(error)
    summary = summary + sum(sum_temp); 

    
% Implement Backpropagation algorithm
 
    d3 = (a_3 .- y_label); % 1x4
    z_1 = [1 z_1]; % size(z_1)=1x26
    d2 = (d3 * Theta2) .* sigmoidGradient(z_1); % ((1x4)x(4x26)).*(1x26)=((1x26).*(1x26))=(1x26)
    d2 = d2(:, 2:end); %removing bais node (1) from begining , now size(d2)=1x25

    Theta1_grad = Theta1_grad + d2' * a_1; %final step :theta1_grad= 25x2501
    Theta2_grad = Theta2_grad + d3' * a_2; %final step :theta2_grad= 4x26

end;

% Add regularization to the cost
regularization = (lambda/(2*m)) * (sum(sum(Theta1(:,2:end) .^ 2, 2)) + sum(sum(Theta2(:,2:end) .^ 2, 2)));

J = (summary/m) + regularization;

% Add regularization to the gradient
Theta1_grad(:,1) = Theta1_grad(:,1) ./ m;
Theta2_grad(:,1) = Theta2_grad(:,1) ./ m;
Theta1_grad(:,2:end) = (Theta1_grad(:,2:end) ./ m) + ((lambda ./ m) * Theta1(:,2:end));
Theta2_grad(:,2:end) = (Theta2_grad(:,2:end) ./ m) + ((lambda ./ m) * Theta2(:,2:end));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
