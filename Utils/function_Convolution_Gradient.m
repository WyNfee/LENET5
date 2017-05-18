%This function will compute the gradient of a convolution process
%input
%p_d: the delta (error to input)
%       this must have been computed out before passing through this
%       function.
%       if we have a covolution input, 3 images with 18 * 18 conved data,
%       each data applies 6 filters
%       the data should be 3 * 1944 (18*18*6) matrix
%   
%p_x: the input data of the current layer
%       if the input of the conved data is 3 image with 20 * 20 resolution
%       then the data should be 3 * 400 (20 * 20) matrix
%p_f_m: filter amount, in comments space, it is 6
%output
%r_grad:
%       the gradient compute based on these parameters
%       in comment space, the filter should be 6 filters with 3 * 3
%       resolutions
function r_grad = function_Convolution_Gradient(p_d, p_x, p_f_m)
    
    % the amount of input;
    t_m = size(p_x, 1);
    %the input size
    t_x_s = size(p_x, 2);
    %the input dimension
    t_x_d = sqrt(t_x_s);
    
    % the size of each delta error to each filter
    t_d_s = size(p_d, 2) /  p_f_m;
    % the delta error for each filter
    t_d_d = sqrt(t_d_s);
    
    %the convolution filter dimension
    t_f_d = t_x_d - t_d_d + 1;
    
    %Worthy to note here, 
    %when computing the gradient, we revert the loop
    %process from compute the convolution, 
    %the reason is we need to compute the gradient for every filter, not
    %every item
   
    %a storage for gradient
    t_grad = [];
   
    %for every filter
    for i = 1 : p_f_m
        
        %a storage for current filter gradient
        %filter grad
        t_f_g = zeros(t_f_d, t_f_d);
        
        %for every input
        for m = 1 : t_m
            %current delta pos
            t_c_d_p = (i - 1) * t_d_s;
            
            %current delta for this filter
            t_c_d = p_d(m,  (t_c_d_p + 1 : t_c_d_p + t_d_s));
            t_c_d = reshape(t_c_d, t_d_d, t_d_d);
            
            %re-org original data
            t_x = p_x(m, :);
            t_x = reshape(t_x, t_x_d, t_x_d);
            
            %compute current filter grad
            t_c_f_g = conv2(t_x, rot90(t_c_d, 2), 'valid');
            %add all grad
            t_f_g = t_f_g + t_c_f_g;
            
        end
        %compute the final grad
        t_f_g = t_f_g ./ t_m;
        %put it into the storage
        
        t_grad = [t_grad; t_f_g(:)];
    end

    % give it to output
    r_grad = t_grad;
    
end