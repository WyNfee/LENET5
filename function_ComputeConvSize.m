function [r_convImageSize, r_valid] = function_ComputeConvSize(p_input_size, p_filter_size ,p_stride_size)
    %assume the input image is square
    %r_convImageSize the compute size of convImageSize
    %valid if stride or filter is correct
    r_convImageSize  = (p_input_size - p_filter_size)/p_stride_size + 1;
    
    r_valid = floor(r_convImageSize) == r_convImageSize;
    
end