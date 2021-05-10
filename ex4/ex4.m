

clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
hidden_layer_size = 30;   % 25 hidden units
num_labels = 26;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('ex4data1.mat');
im = imread("my_drawing.jpg");
im = double(im(:)');  %img(:,:,1) selects the first "pane" of the array "img".double() indicates that the number values of the input should be examined and a new array the same size be constructed which has the same numeric values but represented as double precision data types. 
X = [X; im];
y = [y; 1];
m = size(X, 1);
% Randomly select 100 data points to display
sel = randperm(size(X, 1));
sel = sel(1:100);

%displayData(X(sel, :));

%fprintf('Program paused. Press enter to continue.\n');
%pause;
%exercise 2

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

z=[-1 -0.5 0 0.5 1]
g = 1.0 ./ (1.0 + exp(-z));
fprintf('%f ', g);
fprintf('\n\n');
load('grads.m'); 
%
fprintf('\nprogram paused');
pause;

fprintf('\nTraining Neural Network... \n')
options = optimset('MaxIter', 50);     %returns options with specified parameters set using one or more name-value pair arguments.

%  You should also try different values of lambda
lambda=1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
pred = predict(Theta1, Theta2, X);

predRowSize = size(pred,1);
alphabet = pred(predRowSize);
fprintf('%f',alphabet);
save -text result.txt alphabet
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);



