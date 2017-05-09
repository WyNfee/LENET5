%This function will generate a layer of pooling data by forward propagation
function r_layer_output = function_MaxPoolingLayerForwardPropagation(p_input_data, p_pooling_size)
    
    t_input_data_amount = size(p_input_data, 1);
    t_input_data_size =  sqrt(size(p_input_data, 2));
    t_pooling_data_size = function_ComputeConvSize(t_input_data_size, p_pooling_size, p_pooling_size);
    t_layer_output = zeros(t_input_data_amount, t_pooling_data_size * t_pooling_data_size);
     
     
    for i = 1 : t_input_data_amount
        t_pooling_data = function_MaxPoolingDataGeneration(p_input_data(i,:), p_pooling_size);
        t_layer_output(i, :) = t_pooling_data;
    end
    
    r_layer_output = t_layer_output;
    
end