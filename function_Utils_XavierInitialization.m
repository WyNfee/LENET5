%Use Xavier's method to Initialize the weight
function r_weight = function_Utils_XavierInitialization(p_input_size, p_output_size)
    %need to investigate the what the rand function is usually do, it
    %should be normal distributed
    t_weight = rand(p_output_size,p_input_size);
    t_demoninator = sqrt(p_input_size);
    r_weight = t_weight / t_demoninator;
 
end