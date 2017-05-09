clear; close all; clc;

load('data_MNIST.mat');

g_training_data = X;
g_answer_data = y;
g_training_data_amount = size(X,1);

g_filter_size = 3;
g_filter_stride = 1;
g_filter_amount = 6;

t_pick_data_index = round(rand(1) * g_training_data_amount, 0);

t_pick_data = g_training_data(t_pick_data_index,:);

t_conv_data_Matrix = function_ConvolutionLayerForwardPropagation(t_pick_data, g_filter_size, g_filter_stride, g_filter_amount);

%Let us show the t_conv_data
t_pick_data_size = sqrt(size(t_pick_data, 2));
t_conv_size = function_ComputeConvSize(t_pick_data_size, g_filter_size, g_filter_stride);

t_conv_data_amount = size(t_conv_data_Matrix, 1);

for i = 1: t_conv_data_amount
    t_image_matrix = reshape(t_conv_data_Matrix(i,:), t_conv_size,t_conv_size);
    subplot(3,2,i);
    colormap(gray);
    imagesc(t_image_matrix,  [0 1]);
    axis image off;
end

drawnow;
%**