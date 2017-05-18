%This fucntion do prediction on learnt data
%
function [r_prediction] = function_Do_Prediction...
    (...
    p_x, p_w, ...
    p_st...%the struct
    )
    %unpack the weight
    t_w2_filter_amount = p_st.t_w2_filter_size(1) * p_st.t_w2_filter_size(2);
    t_w2_filter = reshape(p_w(1:t_w2_filter_amount), p_st.t_w2_filter_size);    

    t_w2_bias_pos = t_w2_filter_amount;
    t_w2_bias_amount = p_st.t_w2_bias_size(1) * p_st.t_w2_bias_size(2);
    t_w2_bias = reshape(p_w(t_w2_bias_pos+1 : t_w2_bias_pos + t_w2_bias_amount),p_st.t_w2_bias_size);
    
    t_w3_pos = t_w2_bias_pos + t_w2_bias_amount;
    t_w3_amount = p_st.t_w3_size(1) * p_st.t_w3_size(2);
    t_w3 = reshape(p_w(t_w3_pos+1 : t_w3_pos + t_w3_amount), p_st.t_w3_size);
    
    t_w4_pos = t_w3_pos + t_w3_amount;
    t_w4 = reshape(p_w(t_w4_pos+1 : end), p_st.t_w4_size);
    
    %create the helper variables
    t_m = size(p_x, 1);
    t_helper = ones(t_m, 1);
    
    t_z2 = function_Convolution(p_x, t_w2_filter, t_w2_bias, p_st.t_n_conv_filter);
    t_a2 = function_ReLu(t_z2);
    
    t_a2 = function_MaxPooling2x2(t_a2, p_st.t_n_conv_filter);
    
    %Do hidden 
    t_a2 = [t_helper, t_a2];
    t_z3 = t_a2 * t_w3';
    t_a3 = function_ReLu(t_z3);
    
    %Do output
    t_a3 = [t_helper, t_a3];
    t_z4 = t_a3 * t_w4';
    t_softmax = function_Softmax(t_z4);
    r_prediction = t_softmax';
end