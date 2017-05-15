%Function
%Compute the convolution layer data for forward propagation
%return: r_layer_output:
%is a matrix, with filter_amount * (each cov image matrix)
%return: r_layer_weight
%is a matrix, with filter_amount * (each cov weight matrix)
function [r_layer_output, r_layer_weight]  = function_LENET_ConvolutionLayerForwardPropagation(p_input_data, p_filter_size, p_filter_stride, p_filter_amount)
    %compute the input data size
    t_input_data_size = sqrt(size(p_input_data, 2));

    %init the stored data
    t_conv_data_size = function_Utils_ComputeConvSize(t_input_data_size, p_filter_size, p_filter_stride);
    t_layer_output = zeros(p_filter_amount, t_conv_data_size * t_conv_data_size);
    t_layer_weight = zeros(p_filter_amount, p_filter_size * p_filter_size + 1);

    %generate the conv
    for i = 1 : p_filter_amount
        [t_conv_data, t_conv_weight] = function_LENET_ConvolutionDataGeneration(p_input_data, p_filter_size, p_filter_stride);
        t_layer_output(i,:) = t_conv_data;
        t_layer_weight(i,:) = t_conv_weight;
    end
    
    r_layer_output = t_layer_output;
    r_layer_weight = t_layer_weight;
    
end