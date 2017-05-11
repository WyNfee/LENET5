%This function compte the gradient/derivative of sigmoid activation neuron
function r_sigmoid_gradient = function_Utils_SigmoidGradientFunction(p_input_sigmoid)
    t_sigmoid_gradient = function_Utils_SigmoidFunction(p_input_sigmoid);
    r_sigmoid_gradient = bsxfun(@times, t_sigmoid_gradient ,  (1 - t_sigmoid_gradient));    
end