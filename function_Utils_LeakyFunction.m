%This function return Leaky-Relu signal
%the input data expected not contain the signal of leaky relu
function r_leaky = function_Utils_LeakyFunction(p_input)
    t_leaky_positive = p_input > 0;
    t_leaky_positive = t_leaky_positive .* p_input;
    
    t_leay_negative = p_input <= 0;
    t_leay_negative = t_leay_negative .* 0.01 .* p_input;
    
    r_leaky = t_leaky_positive + t_leay_negative;
end