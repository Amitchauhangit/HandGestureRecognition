%% The required parameters
input_layer_size  = 2500;  % 50x50 Input Images of digits in 0 and 1
hidden_layer_size = 25;   % 25 hidden units
num_labels = 4;          % 4 labels 

% Proccessed image features with 2500 columns for each row
% since there are 2500 pixels (50x50) from every processed image
X_train = dlmread('x_features_train');

X_test = dlmread('x_features_test');

% Labels for each processed training and test image 
%[1 0 0 0] - left, [0 1 0 0] - right, [0 0 1 0] - palm, [0 0 0 1] - peace
y_train = dlmread('y_labels_train');
y_test = dlmread('y_labels_test');

label_keys = { 'left', 'right', 'palm', 'peace'};

% Initialize random weights for start
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);%2500x25
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);%25x4

% Unroll parameters ; starting 62500(2500x25) is initial_Theta1 then 100(25x4) is initial_Theta2
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


options = optimset('MaxIter', 100);
lambda = 1;

% Create the cost function Handler that needs to be minimized
costFunction = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument initial_nn_params(the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options); %Conjugate Gradient function

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

%Prediction based on obtained Theta values
pred = predict(Theta1, Theta2, X_train);

% Compare the prediction with the actual values
[val idx] = max(y_train, [], 2);
fprintf('\nTraining Set Accuracy: %f%%\n', mean(double(pred == idx)) * 100);

% Make the prediction based on obtained Theta values
pred = predict(Theta1, Theta2, X_test);

% Compare the prediction with the actual values
[val idx] = max(y_test, [], 2);
fprintf('\nTest Set Accuracy: %f%%\n', mean(double(pred == idx)) * 100);




test_img = processSkinImage("test/1.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});
pause;

% Predict multiple test images
test_img = processSkinImage("test/2.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/3.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('Type: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/4.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('Type: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/5.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/6.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/7.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});
pause;

test_img = processSkinImage("test/8.jpg");
imshow(test_img);
pred = predict(Theta1, Theta2, test_img(:)')
fprintf('\nType: %s\n\n', label_keys{pred});



