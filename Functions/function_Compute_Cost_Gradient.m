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
    p_w2_filter_size, p_w2_bias_size,...
    p_w3_size, p_w4_size,...
    p_n_conv_filter, p_regularization_param...
    )
    %unpack the weight
    t_w2_filter_amount = p_w2_filter_size(1) * p_w2_filter_size(2);
    t_w2_filter = reshape(p_w(1: t_w2_filter_amount), p_w2_filter_size);
    
    t_w2_bias_pos = t_w2_filter_amount;
    t_w2_bias_amount = p_w2_bias_size(1) * p_w2_bias_size(2);
    t_w2_bias = reshape(p_w(t_w2_bias_pos+1 : t_w2_bias_pos + t_w2_bias_amount), p_w2_bias_size);

    t_w3_pos = t_w2_bias_pos + t_w2_bias_amount;
    t_w3_amount = p_w3_size(1) * p_w3_size(2);
    t_w3 = reshape(p_w(t_w3_pos+1 : t_w3_pos + t_w3_amount), p_w3_size);
    
    t_w4_pos = t_w3_pos + t_w3_amount;
    t_w4 = reshape(p_w(t_w4_pos+1 : end), p_w4_size);
    
    %create a input helper
    t_m = size(p_x, 1);
    t_helper = ones(t_m,1);
    
    %FORWARD PROPAGATION STARTS
    
    %CONVOLUTION FORWARD PROPAGATE STARTS    
    t_z2 = function_Convolution(p_x, t_w2_filter, t_w2_bias, p_n_conv_filter);
    t_a2 = function_ReLu(t_z2);
    
    t_a2 = function_MaxPooling2x2(t_a2, p_n_conv_filter);
    
    %CONVOLUTION FORWARD PROPAGATION END
    
    %FULL CONNECTION LAYER TO HIDDEN FORWARD PROPAGATION START
    
    %prepare the layer one input data
    %add additonal 1 column at the begining 
    t_a2 = [t_helper, t_a2];
    
    %the layer one output
    t_z3 = t_a2 *  t_w3';
        
    %the layer two input
    %add additonal 1 column at the begining
    t_a3 = function_ReLu(t_z3);
    
    %FULL CONNECTION LAYER TO HIDDEN FORWARD PROPAGATION END
    
    %FULL CONNECTION LAYER TO OUTPUT FORWARD PROPAGATION START
    
    t_a3  = [t_helper, t_a3];

    %the layer two output
    t_z4 = t_a3 * t_w4';
    
    %the prediction, the layer 3 data
    t_softmax = function_Softmax(t_z4);
    
    %FULL CONNECTION LAYER TO OUTPUT FORWARD PROPAGATION END
    
    %COMPUTE THE LOST OF THE WHOLE NETWORK START
    
    %compute the error without regularization
    t_cost = function_Softmax_Cost(t_softmax, p_y);
    
    %compute the regularization form
    %regularization form for layer one:
    t_w3_reg = function_NN_Weight_Regularization(t_w3, t_m, p_regularization_param);
    %regularization form for layer two:
    t_w4_reg = function_NN_Weight_Regularization(t_w4, t_m, p_regularization_param);
    
    t_cost = t_cost + t_w3_reg + t_w4_reg;
    
    r_cost = t_cost;
    
    % COMPUTE THE LOST OF THE WHOLE NETWORK END
    
    %FORWARD PROPAGATION END
    
    
    %BACK PROPAGATION START
    
    %COMPUTE FULL CONNECTION TO OUTPUT LAYER ERROR AND GRADIENT START
    %E4/Z4 = E4/softmax * /softmax*Z4 =y-answer
    t_delta_4 = t_softmax - p_y;
   
    %compute the gradient of layer two weight
    %use chain rule to compute: 
    %E3/w2 = E3/a3 * a3/z3 * z3/w2 = 1 * g' * a2 (compute order is not
    %considered)
    t_w4_grad = t_delta_4' * t_a3 / t_m;
    %the bias donot need the regularization
    t_w4_grad_reg = ones(size(t_w4));
    t_w4_grad_reg(:, 1) = 0;
    t_w4_grad_reg = t_w4_grad_reg .* t_w4 * p_regularization_param / t_m;
    t_w4_grad = t_w4_grad + t_w4_grad_reg;
    %COMPUTE FULL CONNECTION TO OUTPUT LAYER ERROR AND GRADIENT END
    
    %COMPUTE FULL CONNECTION TO HIDDEN LAYER ERROR AND GRADIENT START
    %compute the layer 3 error
    t_delta_3 = t_delta_4 * t_w4;
    %compute teh gradient of layer one weight
    %use chain rule as well to compute, but this time, we have to separete
    %the computation, because the hidden layer got 1 addional column, and
    %it shouldnot be put into the gradient
    %transit error to layer one z form E2/Z1 = E2/A2 * A2/Z1
    %but the original output haven't added 1 column, so we add it back
    t_z3 = [t_helper, t_z3];
    %use the changed output data to compute gradient
    t_delta_3 = t_delta_3 .* function_ReLu_Gradient(t_z3);
    %remove the addtional comlumn
    t_delta_3 = t_delta_3(:,(2:size(t_delta_3 ,2)));
    %continue to compute E2/W1 = E2/Z1 * Z1/W1
    t_w3_grad = t_delta_3' *t_a2  / t_m;
    
    %the bias donot need to regularization
    t_w3_grad_reg = ones(size(t_w3));
    t_w3_grad_reg(:,1) = 0;
    t_w3_grad_reg = t_w3_grad_reg .* t_w3 * p_regularization_param / t_m;
    t_w3_grad = t_w3_grad + t_w3_grad_reg;
    %COMPUTE FULL CONNECTION TO HIDDEN LAYER ERROR AND GRADIENT END
    
    %COMPUTE THE CONV FILTER ERROR AND GRADIENT START
    %compute the layer 2 error
    t_delta_2 = t_delta_3 * t_w3;
    %remove the bias column
    t_delta_2 = t_delta_2(:,(2:size(t_delta_2,2)));
    
    t_delta_2 = function_MaxPooling2x2Back(t_delta_2, p_n_conv_filter);
    
    %compute the error
    t_delta_2 = t_delta_2 .* function_ReLu_Gradient(t_z2);


    %Use Conv Operation to compute the grad of conv filter
    t_w2_filter_grad = function_Convolution_Gradient(t_delta_2, p_x, p_n_conv_filter);
 
    %the bias do not need any grad
    t_w2_bias_grad = zeros(p_w2_bias_size);
    
    %COMPUTE THE CONV FILTER ERROR AND GRADIENT END
    
    %pack the gradient again
    r_gradient = [t_w2_filter_grad(:); t_w2_bias_grad(:)];
    r_gradient = [r_gradient(:); t_w3_grad(:)];
    r_gradient = [r_gradient(:) ; t_w4_grad(:)];
    
end