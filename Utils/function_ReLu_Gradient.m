%compute the gradient of the tanh
%in general, p_input should NOT be a Relu signal
function r_gradient = function_ReLu_Gradient (p_input)
    r_gradient  = p_input > 0;
end