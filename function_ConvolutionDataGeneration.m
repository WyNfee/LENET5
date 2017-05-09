function [r_convData, init_weight] = function_ConvolutionDataGeneration(p_input_data, p_filter_size, p_filter_stride)
    %assume that, we all get square data, 
    %and assume the data has format:  1 * element_amount
    t_data_size = sqrt(size(p_input_data, 2));
    
    %compute and init the weight; we expect the filter's size
    t_in_weight_count = p_filter_size * p_filter_size;
    %we only produce 1 number with each filter
    t_out_count = 1;
    % the reason of +1 in the weight initialization, is that we need to
    % implment wx+b, we need left one weight for b
    t_init_weight = function_Utils_XavierInitializationForReLu(t_in_weight_count + 1, t_out_count);
    
   %we assume the stride is valid; won't check it here
   %check method:
   % conv_image_amount_each_edge=
   % (t_data_size - p_filter_size)/p_filter_stride + 1
   %if conv_image_amount_each_edge is not integer
   %then stride is invalid, check function_ComputeConvSize for detail
    t_conv_image_amount_each_edge = function_Utils_ComputeConvSize(t_data_size, p_filter_size, p_filter_stride);
    t_conv_data = zeros(t_conv_image_amount_each_edge,t_conv_image_amount_each_edge);
   %rebuild the data to square matrix
    t_image_data =reshape(p_input_data, [t_data_size,t_data_size]);
    
    t_current_width_index = 0;
    t_current_height_index=  0;
    
    %crop the image from t_image_data, and cov the image 
    %we crop it from left to right, up to down
    while(t_current_width_index < t_conv_image_amount_each_edge)
        while(t_current_height_index < t_conv_image_amount_each_edge)
            
            %this part is trying to locate which place to corp the image,
            %according to stride and filter size
            t_corp_height_start = (t_current_height_index * p_filter_stride)+1;
            t_corp_height_end = t_corp_height_start + p_filter_size - 1;
            
            t_corp_width_start = (t_current_width_index * p_filter_stride)+1;
            t_corp_width_end = t_corp_width_start + p_filter_size - 1;
            
            %corp the image
            t_corped_image_piece = t_image_data(t_corp_height_start:t_corp_height_end, t_corp_width_start: t_corp_width_end);
            
            %reshape the image again to compute with weight
            t_reshaped_data_size = t_in_weight_count;
            t_reshaped_corped_image = reshape(t_corped_image_piece, t_reshaped_data_size, t_out_count);
            
            %because we are doing the ax+b, we need additional 1 place as 1 for b
            t_init_put = ones(t_reshaped_data_size+1, t_out_count);
            t_init_put( 2 : t_reshaped_data_size+1) = t_reshaped_corped_image;
            
            %compute the dot product
            t_conv_data_piece = t_init_weight' * t_init_put;
            
            %put it to conv_data matrix
            t_conv_data(t_current_height_index+1, t_current_width_index+1) = t_conv_data_piece;
            
            t_current_height_index = t_current_height_index + 1;
        end
        t_current_width_index = t_current_width_index + 1;
        t_current_height_index = 0; % reset the height index
    end
    
    r_convData = t_conv_data(:);
    init_weight = t_init_weight;
end