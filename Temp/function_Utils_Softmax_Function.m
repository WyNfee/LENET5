%the softmax operation
%the input data are thoese HAVE NOT been softmax-lized
function r_softmax = function_Utils_Softmax_Function(p_input_data)
    t_exponential = exp(p_input_data);
    t_denominator = sum(t_exponential, 2);
    t_softmax = bsxfun(@rdivide, t_exponential, t_denominator);
    r_softmax = t_softmax;
end
