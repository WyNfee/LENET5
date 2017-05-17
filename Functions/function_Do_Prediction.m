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
    t_x_d =sqrt(size(p_x, 2));
    t_filter_amount = p_st.t_n_conv_filter_size.^2;
    t_filter_size = p_st.t_n_conv_filter_size;
    t_bias_amount = p_st.t_n_conv / p_st.t_n_conv_filter;
    t_bias_size = sqrt(t_bias_amount);
    
    t_z2 = [];
    
    %Do Convolution
    for m = 1 : t_m
        
        t_conved_data = [];      
        
        for i = 1 : p_st.t_n_conv_filter
            %find current filter
            t_current_filter_pos = (i - 1) * t_filter_amount;
            t_current_filter = reshape(t_w2_filter(t_current_filter_pos+1 : t_current_filter_pos + t_filter_amount),t_filter_size, t_filter_size);
            
            %find current filter bias
            t_current_bias_pos = (i-1) * t_bias_amount;
            t_current_bias = reshape(t_w2_bias(t_current_bias_pos+1 : t_current_bias_pos + t_bias_amount), t_bias_size, t_bias_size);
            
            %pick a piece of data to do convolution
            t_data_for_conv = p_x(m,:);
            t_data_for_conv = reshape(t_data_for_conv, t_x_d, t_x_d);
            
            t_data_for_conv = conv2(t_data_for_conv, t_current_filter, 'valid');
            t_data_for_conv = t_data_for_conv + t_current_bias;
            
            t_conved_data = [t_conved_data; t_data_for_conv(:)];
        end
        t_conved_data = t_conved_data';
        t_z2 = [t_z2; t_conved_data];
    end
    
    t_a2 = function_ReLu(t_z2);
    
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