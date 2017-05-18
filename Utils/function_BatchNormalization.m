%do batch normalization for input matrix
function r_bn = function_BatchNormalization(p_x)
    epsilon = 1e-8;
    t_mean_x = mean(mean(p_x));
    t_std_x = std(p_x(:));
    
    r_bn = (p_x - t_mean_x) / (t_std_x + epsilon);
    
end