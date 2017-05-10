%This Script is targeting on using Sigmoid neuron and BGD to do a digit recognition
%It is a full connection neuron network with only 1 hidden layer with 100 hidden neurons and 1 output

%the architecture:
%input: 400 neurons
%hidden layer: 100 neurons, full connection, activation: sigmoid
%output layer: 10 neuron,  full connection, activation: sigmoid

%we don't do batch normalization in architecture for this toy sample

%clear every thing
clear; close all; clc;

%load the data
load('data_MNIST.mat');

%change the variable name of the input data
%we assume the data has been normalized, check the data!
%in practise, we need to do batch normalization
g_input_data = X;

%set up architecture parameters
g_layer_one_neuron_amount = 400;
g_layer_two_neuron_amount = 100;
g_layer_three_neuron_amount = 10;

%build the input answer matrix
g_input_answer_amount = size(y,1);
g_input_answer = zeros(g_input_answer_amount, g_layer_three_neuron_amount);
for i = 1 : g_input_answer_amount
    g_input_answer(i,y(i)) = 1;
end

%init the weight for layer one
g_layer_one_input_size = g_layer_one_neuron_amount + 1;
%g_layer_one_weight = function_Utils_GenericWeigthInitialization(g_layer_one_input_size, g_layer_two_neuron_amount);
g_layer_one_weight = function_Utils_XavierInitialization(g_layer_one_input_size, g_layer_two_neuron_amount);

%init the weight for layer two
g_layer_two_input_size = g_layer_two_neuron_amount + 1;
%g_layer_two_weight = function_Utils_GenericWeigthInitialization(g_layer_two_input_size, g_layer_three_neuron_amount);
g_layer_two_weight = function_Utils_XavierInitialization(g_layer_two_input_size, g_layer_three_neuron_amount);

%pack the weigth together to a BGD process
g_packed_weight = [g_layer_one_weight(:); g_layer_two_weight(:)]; 

%provide the layer one and layer two size
g_layer_one_size = size(g_layer_one_weight);
g_layer_two_size = size(g_layer_two_weight);

%a hyper parameter of regularization param
%roughly set to 0.3, we don't have to dig too much for this reference

g_reularization_param = 0.3;

t_options = optimset('MaxIter', 100);

t_costFunction = @(t_param) function_Ref_CostFunctionSigmoidBGD(t_param,g_input_data, g_input_answer, g_layer_one_size, g_layer_two_size, g_reularization_param);

[t_bgd_weight, t_cost] = function_Utils_ConjunctionGradient(t_costFunction, g_packed_weight, t_options);

%unpack the parameters again 
t_layer_one_weight_size = g_layer_one_size(1) * g_layer_one_size(2);
t_layer_one_weight = reshape(t_bgd_weight ( 1 : t_layer_one_weight_size), g_layer_one_size);
t_layer_two_weight_size = t_layer_one_weight_size+1;
t_layer_two_weight = reshape(t_bgd_weight(t_layer_two_weight_size : end), g_layer_two_size);

%now do the prediction and plot












