%Use Xavier's method to Initialize the weight; only for ReLu
function r_weight = function_XavierInitialization_For_ReLu(p_input_size, p_output_size)
    %need to investigate the what the rand function is usually do, it
    %should be normal distributed
    t_weight = randn(p_output_size,p_input_size);
    t_demoninator = sqrt(2.0 / (p_input_size * p_output_size));
    r_weight = t_weight * t_demoninator;
end