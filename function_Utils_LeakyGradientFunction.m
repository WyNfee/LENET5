%compute the gradient of the tanh
%in general, p_input should NOT be a Leaky-Relu signal
function r_gradient = function_Utils_LeakyGradientFunction(p_input_data)

    t_leaky_grad_positive = p_input_data > 0;
    
    t_leaky_grad_negative = p_input_data <= 0; 
    t_leaky_grad_negative = t_leaky_grad_negative .* 0.01;
    
    r_gradient = t_leaky_grad_positive + t_leaky_grad_negative;

end