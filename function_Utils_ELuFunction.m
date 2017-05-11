%this function return elu signal
function r_elu = function_Utils_ELuFunction(p_input, p_elu_coefficient)
    t_elu_positive = p_input > 0;
    t_elu_positive = t_elu_positive .* p_input;
    
    t_elu_negative = p_input <= 0;
    t_elu_negative = t_elu_negative .* p_elu_coefficient .* (exp(p_input) - 1);
    
    r_elu = t_elu_positive + t_elu_negative;
end