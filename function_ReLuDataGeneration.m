%This function will apply ReLu in batch
%which is faster than apply it one by one
function r_relu_data = function_ReLuDataGeneration(p_input_data)
    %logical operation, return all element 1 if greater than 0, and return
    %0 if less than 0;
    t_relu_data = bsxfun(@gt, p_input_data, 0);
    %mathmatical elementwise multiplication
    r_relu_data = bsxfun(@times, p_input_data, t_relu_data);
end