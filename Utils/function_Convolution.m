%Compute the Convolution data from input and convolution filter group
%input:
%p_x: the input data set, it should be like:
%         say a input is 3 picture with 20 * 20 resolution (only one
%         channel)
%           the input should be 3 * 400 matrix
%p_f: the input filter group, it should be like:
%       say we have a 6 filter with 3 *3 resolution
%           the filter should be 54 * 1 resolution (9 * 9 * 6)
%p_f_b: the filter bias
%       it should the same size of conved data amount
%            say we convlution 20 * 20 image with 6 3 * 3 resolution filter,
%            it should be 1944 * 1 matrix ( 6 * 18 * 18 )
%p_f_m: the filter amount
%return:
%r_conv_data: the conved_data, it should be in this form:
%           say we have convoled data with 18 * 18 resolution for 6
%           filters, we have 3 pictures in total, we will have:
%           3 * 1944 (18 * 18 * 6) matrix
function r_conv_data = function_Convolution(p_x, p_f, p_f_b, p_f_m)
    %the total input data amount
    t_m = size(p_x,1);
    %each input data resolution (assume square)
    t_x_d = sqrt(size(p_x, 2));
    
    %the each filter size
    t_f_s = size(p_f, 1) / p_f_m;
    %each filter data resolution
    t_f_d = sqrt(t_f_s);
    
    %each fiter bias size
    t_f_b_s = size(p_f_b, 1) / p_f_m;
    %each filter bias resolution
    t_f_b_d = sqrt(t_f_b_s);
    
    %a storage to store all conved data
    t_data = [];
    
    %for every input
    for m = 1 : t_m
        %a storage for current cov data
        t_c_data = [];
        
        %for every filter
        for i = 1 : p_f_m
            
            %extract the filter from filter storage
            %current filter position
            t_c_f_p = (i - 1) * t_f_s;
            %extact the filter from filter storage, and reshape to matrix
            %computable
            t_c_f = reshape(p_f(t_c_f_p + 1 : t_c_f_p + t_f_s), t_f_d,t_f_d);
            
            %extract the bias as well (current_filter_bias_position)
            t_c_f_b_p = (i - 1) * t_f_b_s;
            %extract the bias, and reshape to matrix for operation
            t_c_f_b = reshape(p_f_b( t_c_f_b_p + 1 :  t_c_f_b_p + t_f_b_s), t_f_b_d, t_f_b_d);
            
            %reorg the input data
            t_x = p_x(m,:);
            t_x = reshape(t_x, t_x_d, t_x_d);
            
            %convolution
            t_cov = conv2(t_x, t_c_f, 'valid');
            
            %add the bias
            t_cov = t_cov + t_c_f_b;
            
            %put it in the storage
            t_c_data = [t_c_data; t_cov(:)];
            
        end
        
        %reorg data and put into storage
        t_c_data = t_c_data';
        
        t_data = [t_data; t_c_data];
        
    end
    
    %return the result
    r_conv_data = t_data;
    
end