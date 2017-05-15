%This fucntion do prediction on learnt data
%
function [r_prediction_matrix] = function_Do_Prediction...
    (...
    p_input_data, p_learnt_weight, ...
    p_network_struct...
    )
    
    %get network structure info
    t_layer_input_weight_size = p_network_struct.t_layer_input_weight_size;
    t_layer_hidden_weight_size = p_network_struct.t_layer_hidden_weight_size;

    %unpack the parameters again, no matter what happens above, we can still
    %get our descent gradient weight
    t_layer_input_weight_amount = t_layer_input_weight_size(1) * t_layer_input_weight_size(2);
    t_layer_input_weight = reshape(p_learnt_weight ( 1 : t_layer_input_weight_amount), t_layer_input_weight_size);
    t_layer_hidden_weight_amount = t_layer_input_weight_amount+1;
    t_layer_hidden_weight = reshape(p_learnt_weight(t_layer_hidden_weight_amount : end), t_layer_hidden_weight_size);

    %Do prediction
    t_data_amount = size(p_input_data, 1);
    t_helper_for_evaluate = ones(t_data_amount, 1);
    t_input_data_for_evaluate = [t_helper_for_evaluate ,p_input_data];
    t_layer_one_data = function_ReLu(t_input_data_for_evaluate * t_layer_input_weight');
    t_layer_one_data = [t_helper_for_evaluate,t_layer_one_data];
    t_predictions_matrix = function_Softmax(t_layer_one_data * t_layer_hidden_weight');
    t_predictions_matrix = t_predictions_matrix';
    
    %return the prediction
    r_prediction_matrix = t_predictions_matrix;
    

end