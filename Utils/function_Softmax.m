%the softmax operation
%the input data are thoese HAVE NOT been softmax-lized
%use the trick to make the softmax stable:)
function r_softmax = function_Softmax(p_input_data)
    t_input_data = p_input_data - max(p_input_data, [] , 2);
    t_exponential = exp(t_input_data);
    t_denominator = sum(t_exponential, 2);
    t_softmax = bsxfun(@rdivide, t_exponential, t_denominator);
    r_softmax = t_softmax;
end
