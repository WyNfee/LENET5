addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Temp'));


image = imread('test2.bmp');
image_data = im2double(image);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
image_data = ((image_data(:,:,1) + image_data(:,:,2) + image_data(:,:,3)) / 3.0);

image_data_centered = (image_data - mean(image_data)) ./std(image_data);

conv_filter = function_XavierInitialization_For_ReLu(3,3);

conv_filter = rot90(conv_filter, 2);

conv_data = conv2(image_data_centered, conv_filter, 'valid');

conv_data = conv_data(:);

conv_bias = function_XavierInitialization_For_ReLu(size(conv_data, 2), size(conv_data, 1));

conv_data  = conv_bias + conv_data;

conv_data =reshape(conv_data, sqrt(size(conv_data,1)),sqrt(size(conv_data,1)));

conv_data = function_ReLu(conv_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conv_filter = function_XavierInitialization_For_ReLu(3,3);

conv_filter = rot90(conv_filter, 2);

conv_data2 = conv2(image_data_centered, conv_filter, 'valid');

conv_data2 = conv_data2(:);

conv_bias = function_XavierInitialization_For_ReLu(size(conv_data2, 2), size(conv_data2, 1));

conv_data2 = conv_data2 + conv_bias;

conv_data2 =reshape(conv_data2, sqrt(size(conv_data2,1)),sqrt(size(conv_data2,1)));

conv_data2 = function_ReLu(conv_data2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

conv_filter = function_XavierInitialization_For_ReLu(3,3);

conv_filter = rot90(conv_filter, 2);

conv_data3 = conv2(image_data_centered, conv_filter, 'valid');

conv_data3 = conv_data3(:);

conv_bias = function_XavierInitialization_For_ReLu(size(conv_data3, 2), size(conv_data3, 1));

conv_data3 = conv_data3 + conv_bias;

conv_data3 =reshape(conv_data3, sqrt(size(conv_data3,1)),sqrt(size(conv_data3,1)));

conv_data3 = function_ReLu(conv_data3);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(2,2,1);
%colormap(gray);
imagesc(image_data);
axis image off;

subplot(2,2,2)
%colormap(gray);
imagesc(conv_data2);
axis image off;

subplot(2,2,3)
%colormap(gray);
imagesc(conv_data3);
axis image off;

subplot(2,2,4)
%colormap(gray);
imagesc(conv_data);
axis image off;