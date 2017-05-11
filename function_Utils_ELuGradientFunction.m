%this function return the gradient of elu function
%the p_input expect not to be a elu signal

function r_gradient = function_Utils_ELuGradientFunction(p_input_data, p_elu_coefficient)
    t_elu_grad_positive = p_input_data > 0;
    
    t_elu_grad_negative = p_input_data <= 0;
    t_elu_grad_negative = t_elu_grad_negative .* p_elu_coefficient .* exp(p_input_data);
    
    r_gradient = t_elu_grad_positive + t_elu_grad_negative;
end