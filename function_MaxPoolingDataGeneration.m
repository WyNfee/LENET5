%Use Max Pooling Mehtod to compute the pooling(downsample) image
function r_pooling_data = function_MaxPoolingDataGeneration(p_input_data, p_pooling_size)
    %compute the input data size
    t_input_data_size = sqrt(size(p_input_data,2));

    %will not perform the valid check here
    %assume that the size will always match
    %we can also using CompteCovSize to do this check, but this time, the
    %stride is the p_pooling_size
    t_pooling_data_size = function_ComputeConvSize(t_input_data_size, p_pooling_size, p_pooling_size);
    %Init the pooling data;
    t_pooling_data = zeros(t_pooling_data_size, t_pooling_data_size);
    
    %reshape the input data for pooling
    t_reshape_input_data = reshape(p_input_data, [t_input_data_size,t_input_data_size]);
    
    t_current_width_index = 0;
    t_current_height_index= 0;
    
    %crop the image from the t_reshape_input_data, to pooling operation
    %from left to right, up to down
    while(t_current_width_index < t_pooling_data_size)
        while(t_current_height_index < t_pooling_data_size)
            
            %try to locate where to corp the image matrix
            t_corp_height_start = (t_current_height_index * p_pooling_size) + 1;
            t_corp_height_end = t_corp_height_start + p_pooling_size - 1;
            
            t_corp_width_start = (t_current_width_index * p_pooling_size) + 1;
            t_corp_width_end = t_corp_width_start + p_pooling_size - 1;
            
            %crop the data
            t_corped_input_data = t_reshape_input_data(t_corp_height_start : t_corp_height_end, t_corp_width_start : t_corp_width_end);
            
            %because the t_corped_input_data is a matrix (not tensor)
            %twice max will give us the max value among them
            %no matter how big they are
            t_pooling_data_piece = max(max(t_corped_input_data));
            
            %put the data piece into the data matrix
            t_pooling_data(t_current_height_index + 1, t_current_width_index + 1) = t_pooling_data_piece;
            
            t_current_height_index = t_current_height_index+1;
        end
        
        t_current_width_index = t_current_width_index + 1;
        t_current_height_index = 0;%reset the height index
    end
    
    %pack the data and output
    r_pooling_data = t_pooling_data(:);
    
end