clear; close all; clc;

load('data_MNIST.mat');

g_training_data = X;
g_answer_data = y;
g_training_data_amount = size(X,1);

g_filter_size = 3;
g_filter_stride = 1;

t_pick_data_index = round(rand(1) * g_training_data_amount, 0);

t_pick_data = g_training_data(t_pick_data_index,:);

t_conv_data = function_ConvolutionData(t_pick_data, g_filter_size, g_filter_stride);
t_conv_data_2 = function_ConvolutionData(t_pick_data, g_filter_size, g_filter_stride);

%Let us show the t_conv_data
t_pick_data_size = sqrt(size(t_pick_data, 2));
t_conv_size = function_ComputeConvSize(t_pick_data_size, g_filter_size, g_filter_stride);

t_image_matrix = reshape(t_conv_data, t_conv_size,t_conv_size);
subplot(1,2,1);
colormap(gray);
imagesc(t_image_matrix,  [0 1]);
axis image off;

t_image_matrix_2 = reshape(t_conv_data_2, t_conv_size,t_conv_size);
subplot(1,2,2);
colormap(gray);
imagesc(t_image_matrix_2,  [0 1]);
axis image off;

drawnow;