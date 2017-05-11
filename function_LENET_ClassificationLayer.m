function r_prediction = function_LENET_ClassificationLayer(p_input_data, p_answer)
    t_softmax = function_Utils_Softmax_Function(p_input_data);
    r_prediction = t_softmax(p_answer);
end