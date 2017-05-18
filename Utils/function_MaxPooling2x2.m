%This function is doing pooling operation, by 2 * 2
%input
%p_x: the data need to do pooling
%output:
%r_p: the data after pooling
function r_pooling = function_MaxPooling2x2(p_x, p_f_m)

    %the size of the input x;
    t_m = size(p_x, 1);
    t_x_s = size(p_x,2) / p_f_m;
    t_x_d = sqrt(t_x_s);
    
    %a storage for pooled data
    t_pool = [];
    
    for m = 1 : t_m
        %current data
        t_data = p_x(m, :);
        
        %a storage for current data
        t_data_pool=[];
        
        for i = 1 : p_f_m
            %current data position
            t_c_d_p = (i - 1) * t_x_s;
            
            %grab the current data
            t_c_data = t_data( (t_c_d_p + 1) : (t_c_d_p + t_x_s));
            t_c_data = reshape( t_c_data, t_x_d,t_x_d);
            
            %do pooling
            %iteration indicator
            t_iter = floor((t_x_d+1)/2);
            
            %current pooling
            t_c_p = zeros(t_iter, t_iter);
            
            %height index
           for h = 1 : t_iter
               %width index
               for w = 1 : t_iter
                   
                   %compute each index
                   t_h = (h -1) * 2;
                   t_w = (w - 1) * 2;
                   
                   %reshape to image
                   t_p_data = reshape(t_c_data (t_w + 1 : t_w+2, t_h+1:t_h+2), 2, 2);
                   %do pooling
                   t_c_p(w, h) = max(max(t_p_data));
                   
                   
               end
           end
           %store the pooling data
           t_data_pool = [t_data_pool; t_c_p(:)];           
        end
        
        t_data_pool = t_data_pool';
        t_pool = [t_pool; t_data_pool];
    end
    
    
    r_pooling = t_pool;
end