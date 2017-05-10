%This function will compute the overall error of the sigmoid function
%No Regularization form involved
function r_cost = function_Utils_SigmoidCost(p_input_data, p_answer_data, p_input_data_amount)
    t_positive_score = bsxfun(@times, log(p_input_data), -p_answer_data);
    t_negative_score = bsxfun(@times, log(1-p_input_data), -(1-p_answer_data));
    
    t_error = t_positive_score + t_negative_score;
    
    r_cost = sum(sum(t_error)) / p_input_data_amount;
    
end