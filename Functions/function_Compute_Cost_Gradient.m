%the core learning argorithm of Batch Gradient Desent with Conjunction Gradient Descent
%the neuron activation is using ReLu
%this including both forward propagation and back propagation
%param:
%p_input_weight: the weight used in this learning process
%p_input_data: the input data used in this learning process
%p_layer_input_weight_size: the weight size of input
function [r_cost, r_gradient] = function_Compute_Cost_Gradient...
    (...
    p_input_weight, p_input_data, p_answer_data, ...
    p_layer_input_weight_size, p_layer_hidden_weight_size, ...
    p_regularization_param...
    )
    %unpack the weight
    t_layer_input_weight_size = p_layer_input_weight_size(1) * p_layer_input_weight_size(2);
    t_layer_input_weight = reshape(p_input_weight(1 :   t_layer_input_weight_size), p_layer_input_weight_size);
    t_layer_hidden_weight_size = t_layer_input_weight_size+1;
    t_layer_hidden_weight = reshape(p_input_weight(t_layer_hidden_weight_size : end), p_layer_hidden_weight_size);
    
    %create a input helper
    t_size_input_data = size(p_input_data, 1);
    t_input_helper = ones(t_size_input_data,1);
    
    %prepare the layer one input data
    %add additonal 1 column at the begining 
    t_layer_one_input_data = [t_input_helper, p_input_data];
    
    %the layer one output
    t_layer_one_output_data = t_layer_one_input_data *  t_layer_input_weight';
    
    %the layer two input
    %add additonal 1 column at the begining
    t_layer_two_input_data = function_ReLu(t_layer_one_output_data);
    t_layer_two_input_data  = [t_input_helper, t_layer_two_input_data];

    %the layer two output
    t_layer_two_output_data = t_layer_two_input_data * t_layer_hidden_weight';
    
    %the prediction, the layer 3 data
    t_layer_three_data = function_Softmax(t_layer_two_output_data);
    
    %compute the error without regularization
    t_cost_of_network = function_Softmax_Cost(t_layer_three_data, p_answer_data);
    
    %compute the regularization form
    %regularization form for layer one:
    t_layer_one_weight_reg = function_NN_Weight_Regularization(t_layer_input_weight, t_size_input_data, p_regularization_param);
    %regularization form for layer two:
    t_layer_two_weight_reg = function_NN_Weight_Regularization(t_layer_hidden_weight, t_size_input_data, p_regularization_param);
    
    t_cost_of_network = t_cost_of_network + t_layer_one_weight_reg + t_layer_two_weight_reg;
    
    r_cost = t_cost_of_network;
    
    %forward propagation complete and get the error
    
    
    %start to use back propagation to get the gradient
    
    %compute the layer 3 error
    t_layer_three_error = t_layer_three_data - p_answer_data;

    %compute the layer 2 error
   t_layer_two_error = t_layer_three_error * t_layer_hidden_weight;
   
    %compute the gradient of layer two weight
    %use chain rule to compute: 
    %E3/w2 = E3/a3 * a3/z3 * z3/w2 = 1 * g' * a2 (compute order is not
    %considered)
    t_layer_two_weight_gradient = t_layer_three_error' * t_layer_two_input_data / t_size_input_data;
    %the bias donot need the regularization
    t_layer_two_weight_gradient_regularizedform = ones(size(t_layer_hidden_weight));
    t_layer_two_weight_gradient_regularizedform(:, 1) = 0;
    t_layer_two_weight_gradient_regularizedform = t_layer_two_weight_gradient_regularizedform .* t_layer_hidden_weight * p_regularization_param / t_size_input_data;
    t_layer_two_weight_gradient = t_layer_two_weight_gradient + t_layer_two_weight_gradient_regularizedform;
    
    %compute teh gradient of layer one weight
    %use chain rule as well to compute, but this time, we have to separete
    %the computation, because the hidden layer got 1 addional column, and
    %it shouldnot be put into the gradient
    %transit error to layer one z form E2/Z1 = E2/A2 * A2/Z1
    %but the original output haven't added 1 column, so we add it back
    t_layer_one_output_data = [t_input_helper, t_layer_one_output_data];
    %use the changed output data to compute gradient
    t_layer_one_error = t_layer_two_error .* function_ReLu_Gradient(t_layer_one_output_data);
    %remove the addtional comlumn
    t_layer_one_error = t_layer_one_error(:,(2:size(t_layer_one_error ,2)));
    %continue to compute E2/W1 = E2/Z1 * Z1/W1
    t_layer_one_weight_gradient = t_layer_one_error' *t_layer_one_input_data  / t_size_input_data;
    
    %the bias donot need to regularization
    t_layer_one_weight_graident_gregularizationform = ones(size(t_layer_input_weight));
    t_layer_one_weight_graident_gregularizationform(:,1) = 0;
    t_layer_one_weight_graident_gregularizationform = t_layer_one_weight_graident_gregularizationform .* t_layer_input_weight * p_regularization_param / t_size_input_data;
    t_layer_one_weight_gradient = t_layer_one_weight_gradient + t_layer_one_weight_graident_gregularizationform;
    
    
    %pack teh gradient again
    r_gradient = [t_layer_one_weight_gradient(:) ; t_layer_two_weight_gradient(:)];
    
end