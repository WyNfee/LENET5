%This function will return relu
%the p_input parameter are not tahn signal
function r_relu = function_Utils_ReLuFunction(p_input)
    t_relu = p_input > 0;
    r_relu = t_relu .* p_input;
end