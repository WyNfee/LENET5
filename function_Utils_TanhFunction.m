%This function will return tanh
%the p_input parameter are not tahn signal
function r_tanh = function_Utils_TanhFunction (p_input)
    t_param1 = exp(p_input);
    t_param2 = exp(-p_input);
    t_numerator =  t_param1 - t_param2;
    t_denominator = t_param1 +  t_param2;
    r_tanh = t_numerator ./ t_denominator;
end