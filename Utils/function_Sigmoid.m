%This function is a sigmoid function
function r_sigmoid = function_Sigmoid(p_input)
    t_denominator = 1 + exp(-p_input);
    r_sigmoid = bsxfun(@rdivide, 1, t_denominator);
end