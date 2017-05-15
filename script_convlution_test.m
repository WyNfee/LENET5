addpath(genpath('Data'))

load('data_MNIST.mat');

picked_index = floor(rand(1) * 1000);
picked_data = test_data(picked_index,:);

picked_data = reshape(picked_data, 20, 20);

conv_filter = function_XavierInitialization_For_ReLu(3,3);

conv_filter = rot90(conv_filter, 2);

conv_data = conv2(picked_data, conv_filter);

subplot(2,1,1);
imagesc(conv_data);
axis image off;

subplot(2,1,2)
imagesc(picked_data);
axis image off;
