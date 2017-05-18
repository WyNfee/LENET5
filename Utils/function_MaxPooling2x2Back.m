%This function do upsampling operation, revert the image back from pooling
%input:
%p_x: the input of matrix need to revert/upsampling
%p_f_m; the filter amount
%output
%r_upsampling: the upsampled data
function r_upsampling = function_MaxPooling2x2Back(p_x, p_f_m)
    %the data amount
    t_m = size(p_x, 1);
    t_x_s = size(p_x, 2)/p_f_m;
    t_x_d = sqrt(t_x_s);
    
    %the storage for upsample data
    t_upsampling = [];
    
    for m = 1 : t_m
        %for our current upsampling data
        t_c_upsampling = [];
        
        t_data = p_x(m, :);
        
        for i = 1 : p_f_m
            
            %current data position
            t_c_d_p = (i - 1) * t_x_s;
            
            %extract the data
            t_c_data = reshape(t_data(t_c_d_p + 1 : t_c_d_p + t_x_s), t_x_d, t_x_d);
            
            %a container for this upsample process
            t_c_up_sample = zeros(t_x_d * 2, t_x_d * 2);
            
            for h = 1 : t_x_d
                for w = 1 : t_x_d
                    
                    t_elment = t_c_data(h, w);
                    
                    t_h = h * 2 - 1;
                    t_w = w * 2 -1;
                    
                    %up sampling
                    t_c_up_sample(t_h : t_h+1, t_w : t_w+1) = t_elment;
                    
                end
            end
            %store the sample data
            t_c_upsampling = [t_c_upsampling; t_c_up_sample(:)];
        end
        %put it to upsample container
        t_c_upsampling = t_c_upsampling';
        t_upsampling = [t_upsampling; t_c_upsampling];
    end
    r_upsampling = t_upsampling;
end