%Use Xavier's method to Initialize the weight; only for ReLu
function r_weight = function_Utils_XavierInitializationForReLu(p_input_size, p_output_size)
    %need to investigate the what the rand function is usually do, it
    %should be normal distributed
    t_weight = rand(p_input_size,p_output_size);
    t_demoninator = sqrt(p_input_size/2);
    r_weight = t_weight / t_demoninator;
 
end