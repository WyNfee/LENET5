function r_softmax = function_Utils_Softmax(p_input_data)
    t_exponential = exp(p_input_data);
    t_denominator = sum(t_exponential);
    t_softmax = bsxfun(@rdivide, t_exponential, t_denominator);
    r_softmax = t_softmax;
end
