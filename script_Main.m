clear; close all; clc;

load('data_MNIST.mat');

g_training_data = X;
g_answer_data = y;
g_training_data_amount = size(X,1);

g_filter_size = 3;
g_filter_stride = 1;
g_filter_amount = 6;

g_max_pooling_size = 2;

t_pick_data_index = round(rand(1) * g_training_data_amount, 0);

t_pick_data = g_training_data(t_pick_data_index,:);

t_conv_data_Matrix = function_ConvolutionLayerForwardPropagation(t_pick_data, g_filter_size, g_filter_stride, g_filter_amount);
t_relu_data_Matrix = function_ReLuLayerForwardPropagation(t_conv_data_Matrix);
t_pooling_data_Matrix = function_MaxPoolingLayerForwardPropagation(t_relu_data_Matrix, g_max_pooling_size);

%Let us show the t_conv_data

t_conv_size = sqrt(size(t_relu_data_Matrix, 2));
t_pooling_size = sqrt(size(t_pooling_data_Matrix, 2));

t_conv_data_amount = size(t_relu_data_Matrix, 1);

figure('Name', 'Conv Data');

for i = 1: t_conv_data_amount
    t_image_matrix = reshape(t_relu_data_Matrix(i,:), t_conv_size,t_conv_size);
    subplot(3,2,i);
    colormap(gray);
    imagesc(t_image_matrix,  [0 1]);
    axis image off;
end

drawnow;

figure('Name', 'Pooling Data');

for i = 1: t_conv_data_amount
    t_image_matrix = reshape(t_pooling_data_Matrix(i,:), t_pooling_size,t_pooling_size);
    subplot(3,2,i);
    colormap(gray);
    imagesc(t_image_matrix,  [0 1]);
    axis image off;
end

drawnow;
%**