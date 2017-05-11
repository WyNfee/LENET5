% the cost function of softmax
% return the cost of the softmax function
%p_input_data is the data we compute for softmax, which should has been softmax-lized 
% the answer data should contain the accurate class information
% say, a 3 classes, and the second one is the class for a specific cases
% the p_answer_data should be [0, 1, 0]
function r_cost = function_Utils_SoftmaxCost (p_input_data, p_answer_data)

    %first, compute the cost
    t_cost = log(p_input_data);
    %second, remove the terms that should not take part in to cost computation
    % because other place of anwse is 0, it can be easily removed by
    % element-wise multiply
    t_cost  = t_cost .* p_answer_data;
    
    %compute the sum
    r_cost = -sum(sum(t_cost));

end