%This function is used to preprocess the data for image machine leanring
%the idea is: center and normalize
%param:
%p_input_data: the input data to do preprocessing
%return:
%r_data: the data used for image processing
function r_data = function_Preprocess_Data(p_input_data)
    %compute the mean of the data
    t_mean_data = mean(p_input_data, 2);
    
    %zero center these data
    t_input_data = double(p_input_data);
    t_pre_proc_data = bsxfun(@minus, t_input_data, t_mean_data);
    
    %normalize these data
    %1. find the absolute max one in each image
    t_absolute_max_element = (max(abs(t_pre_proc_data), [], 2));
    %2. compute the data, and squash the data into [-1, 1]
    r_data = t_pre_proc_data ./ t_absolute_max_element;
    
end