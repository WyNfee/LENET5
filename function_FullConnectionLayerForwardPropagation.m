%This is full connection layer, where connect to all hidden layers of
%generic neuron network
function [r_layer_output, r_layer_weight] = function_FullConnectionLayerForwardPropagation(p_input_data, p_hidden_neuron_amount)
    %unpack the matrix here
    t_input_data = p_input_data(:);
    
    %get size information
    t_input_data_size = size(t_input_data, 1);
    
    %the actual size we need
    t_neural_amount = t_input_data_size + 1;
    
    %t_init_data should be 
    t_init_data = ones(t_neural_amount,1);
    t_init_data(2:t_neural_amount) = t_input_data;
    
    %initialize the weight
    t_init_weight = function_Utils_XavierInitializationForReLu(t_neural_amount, p_hidden_neuron_amount);
    
    r_layer_output = t_init_weight'  * t_init_data;
    
    r_layer_weight = t_init_weight;
end