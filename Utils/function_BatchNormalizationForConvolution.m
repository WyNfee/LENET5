%specially compute batch normalization for convolution layer
%we need to compute the bn for each filter
function r_bn = function_BatchNormalizationForConvolution(p_x, p_n_f)

    t_x_s = size(p_x ,2)/p_n_f;
    
    t_bn = [];
    
    for i = 1 : p_n_f
        
        %data position 
        t_d_p = (i - 1) * t_x_s;
        
        t_batch_data = p_x(:, (t_d_p + 1 : t_d_p+ t_x_s));
        
        t_batch_data = function_BatchNormalization(t_batch_data);
        
        t_bn = [t_bn, t_batch_data];
        
    end
    
    r_bn = t_bn;

end