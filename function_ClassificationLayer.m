function r_prediction = function_ClassificationLayer(p_input_data, p_answer)
    t_softmax = function_Utils_Softmax(p_input_data);
    r_prediction = t_softmax(p_answer);
end