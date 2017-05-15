function function_Check_Performance(p_input_data, p_answer_data, p_learnt_weight, p_network_struct)
    

   %Do prediction
   t_predictions_matrix = function_Do_Prediction(p_input_data, p_learnt_weight,p_network_struct);
    %we use the max probability in k-means output, in practise, sometimes using
    %top 5 output, this cases is so small, using top 5 is silly
    [t_probability, t_prediction] = max(t_predictions_matrix);

    %output the result
    t_right_prediction_count = sum(t_prediction' == p_answer_data);
    t_accuracy = t_right_prediction_count / size(p_answer_data,1);

    fprintf('prediction accurracy %1.6f\n',t_accuracy)
end