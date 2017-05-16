%the core learning argorithm of Batch Gradient Desent with Conjunction Gradient Descent
%the neuron activation is using ReLu
%this including both forward propagation and back propagation
%param:
%p_input_weight: the weight used in this learning process
%p_input_data: the input data used in this learning process
%p_layer_input_weight_size: the weight size of input
function [r_cost, r_gradient] = function_Compute_Cost_Gradient...
    (...
    p_w, p_x, p_y, ...
    p_w2_size, p_w3_size, ...
    p_regularization_param...
    )
    %unpack the weight
    t_w2_amount = p_w2_size(1) * p_w2_size(2);
    t_w2 = reshape(p_w(1 :   t_w2_amount), p_w2_size);
    t_w3_pos = t_w2_amount+1;
    t_w3 = reshape(p_w(t_w3_pos : end), p_w3_size);
    
    %create a input helper
    t_m = size(p_x, 1);
    t_helper = ones(t_m,1);
    
    %prepare the layer one input data
    %add additonal 1 column at the begining 
    t_x = [t_helper, p_x];
    
    %the layer one output
    t_z2 = t_x *  t_w2';
    
    %the layer two input
    %add additonal 1 column at the begining
    t_a2 = function_ReLu(t_z2);
    t_a2  = [t_helper, t_a2];

    %the layer two output
    t_z3 = t_a2 * t_w3';
    
    %the prediction, the layer 3 data
    t_softmax = function_Softmax(t_z3);
    
    %compute the error without regularization
    t_cost = function_Softmax_Cost(t_softmax, p_y);
    
    %compute the regularization form
    %regularization form for layer one:
    t_w2_reg = function_NN_Weight_Regularization(t_w2, t_m, p_regularization_param);
    %regularization form for layer two:
    t_w3_reg = function_NN_Weight_Regularization(t_w3, t_m, p_regularization_param);
    
    t_cost = t_cost + t_w2_reg + t_w3_reg;
    
    r_cost = t_cost;
    
    %forward propagation complete and get the error
    
    
    %start to use back propagation to get the gradient
    
    %compute the layer 3 error; the 
    %E3/Z3 = E3/softmax * /softmax*Z3 =y-answer
    t_delta_3 = t_softmax - p_y;

    %compute the layer 2 error
   t_delta_2 = t_delta_3 * t_w3;
   
    %compute the gradient of layer two weight
    %use chain rule to compute: 
    %E3/w2 = E3/a3 * a3/z3 * z3/w2 = 1 * g' * a2 (compute order is not
    %considered)
    t_w3_grad = t_delta_3' * t_a2 / t_m;
    %the bias donot need the regularization
    t_w3_grad_reg = ones(size(t_w3));
    t_w3_grad_reg(:, 1) = 0;
    t_w3_grad_reg = t_w3_grad_reg .* t_w3 * p_regularization_param / t_m;
    t_w3_grad = t_w3_grad + t_w3_grad_reg;
    
    %compute teh gradient of layer one weight
    %use chain rule as well to compute, but this time, we have to separete
    %the computation, because the hidden layer got 1 addional column, and
    %it shouldnot be put into the gradient
    %transit error to layer one z form E2/Z1 = E2/A2 * A2/Z1
    %but the original output haven't added 1 column, so we add it back
    t_z2 = [t_helper, t_z2];
    %use the changed output data to compute gradient
    t_delta_2 = t_delta_2 .* function_ReLu_Gradient(t_z2);
    %remove the addtional comlumn
    t_delta_2 = t_delta_2(:,(2:size(t_delta_2 ,2)));
    %continue to compute E2/W1 = E2/Z1 * Z1/W1
    t_w2_grad = t_delta_2' *t_x  / t_m;
    
    %the bias donot need to regularization
    t_w2_grad_reg = ones(size(t_w2));
    t_w2_grad_reg(:,1) = 0;
    t_w2_grad_reg = t_w2_grad_reg .* t_w2 * p_regularization_param / t_m;
    t_w2_grad = t_w2_grad + t_w2_grad_reg;
    
    
    %pack teh gradient again
    r_gradient = [t_w2_grad(:) ; t_w3_grad(:)];
    
end