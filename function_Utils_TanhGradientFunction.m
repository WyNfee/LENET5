%compute the gradient of the tanh
%in general, p_input should NOT be a tahn signal
function r_gradient = function_Utils_TanhGradientFunction(p_input)
    t_tanh_gradient = function_Utils_TanhFunction(p_input);
    %should be element wise power 2
    r_gradient = 1 - (t_tanh_gradient.^2);
end