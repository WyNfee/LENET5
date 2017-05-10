%This function compute the weight regularization form of a sigmoid function
%accept a 2D matrix of weight
function r_regularized_form = function_Utils_SigmoidWeightRegularization(p_input_weight, p_input_data_amount, p_regularization_param)
    %prepare the weight, we should not compute the bias (the first row)
    %Need to understand why we remove the first column:  
    
    
    %first we add before to regularization
    t_input_weight = p_input_weight;
    
    t_input_weight(:, 1) = 0;
    
    %now compute the parameter
    t_regularized_weight = bsxfun(@power, t_input_weight, 2);
    t_overall_error = sum(sum(t_regularized_weight));
    r_regularized_form = t_overall_error * p_regularization_param / (2 * p_input_data_amount);
end