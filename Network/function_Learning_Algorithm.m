%the core learning algorithm locate
%param:
%p_training_data: the training data set
%p_training_answer: the training answer
%output
%r_learnt_weight: the weight learnt in this training data set
%r_cost_history: the cost history used to plot learning performance
%r_network_struct: the information about this learning algorithm
function[r_learnt_weight, r_cost_history, r_network_struct] = function_Learning_Algorithm(p_training_data, p_y)
    %set up architecture parameters
    t_n_input = 400;
    t_n_conv_filter = 6;
    t_n_conv_filter_size = 3;
    %conv square matrix (20) with filter/ kernel square matrix (3) with valid command, 
    %will generate square matrix 18*18, and we expect the have 6 layer of filters/kernels 
    %we expect to have 18*18*6 neuron as output of conv operation
    %the computer tation is below:
    t_n_conv = (sqrt(t_n_input) - t_n_conv_filter_size + 1).^2 * t_n_conv_filter;  
    t_n_fc_hidden = 100;
    t_n_output = 10;
    
    
    %reorganize the answer data
    t_n_all_data = size(p_y, 1);
    
    %the w2 filter
    t_init_w2_filter = [];
    %init all conv filters
    for i = 1 : t_n_conv_filter
        %conv filter must init with Gaussian Distribution data
        %so we MUST use function like randn NOT rand
        t_conv_w_filter = function_XavierInitialization_For_ReLu(t_n_conv_filter_size, t_n_conv_filter_size);
        t_init_w2_filter = [t_init_w2_filter; t_conv_w_filter(:)];
    end
    
    %the w2 bias
    t_init_w2_bias = [];
    %the bias size
    t_w2_bias_size = sqrt(t_n_input) - t_n_conv_filter_size + 1;
    for i = 1 : t_n_conv_filter
        t_conv_w_bias = function_XavierInitialization_For_ReLu(t_w2_bias_size, t_w2_bias_size);
        t_init_w2_bias = [t_init_w2_bias;t_conv_w_bias(:)];
    end
    
    %after process, init 2 should be a (n * 1) matrix, 
    %and each (t_n_conv_filter_size * t_n_conv_filter_size) is normal distribution
    %when pack it out, it should be carefully considered
    
    %init the weight for layer one
    t_init_w3 = function_XavierInitialization_For_ReLu(t_n_conv + 1, t_n_fc_hidden);

    %init the weight for layer two
    t_init_w4 = function_XavierInitialization_For_ReLu( t_n_fc_hidden + 1, t_n_output);

    %pack the weigth together to compute the weight
    t_packed_init_w = [t_init_w2_filter(:); t_init_w2_bias(:)];
    t_packed_init_w = [t_packed_init_w(:); t_init_w3(:)];
    t_packed_init_w = [t_packed_init_w(:); t_init_w4(:)]; 

    %provide the layer one and layer two size
    t_w2_filter_size = size(t_init_w2_filter);
    t_w2_bias_size = size(t_init_w2_bias);
    t_w3_size = size(t_init_w3);
    t_w4_size = size(t_init_w4);

    %a hyper parameter of regularization param, close the regularization here
    t_reg_param = 0.1;

    %assign the hyperparameter learning rate
    t_learning_rate = 0.01;
        
    %learning rate decay param
    t_learining_rate_decay_frequency = 800;
    
    %learning rate decay ratio
    t_learning_rate_decay_ratio = 0.8;

    %assgin the hyperparameter, stochasic data size
    t_stochasitic_data_size = 50;

    %a storage for weigth in sgd 
    t_learnt_w  = t_packed_init_w;

    %define the iteration time, 1000 * 100 times
    t_iteration_time = 10000;

    %record frequency
    t_record_frequency = 100;

    %a cost data storage for further ploting
    t_record_cost_data = zeros(t_iteration_time/t_record_frequency, 1);

    %adam param
    t_adam_param_epsilon = 1e-8;
    t_adam_param_beta1 = 0.9;
    t_adam_param_beta2 = 0.9999;

    t_adam_weight_movement = zeros(size(t_learnt_w));
    t_adam_weight_velocity = zeros(size(t_learnt_w));
    
    %do gradient descent
    for i = 1: t_iteration_time
        
        %this will make the data always within [1:g_input_answer_amount]
        t_picked_index = floor(rand(1, t_stochasitic_data_size) * (t_n_all_data - 1) ) + 1;
        %generate stocastic data from dataset
        t_picked_X = p_training_data(t_picked_index(1:end), :);
        %provide the coresponding answer data as well
        t_picked_y = p_y(t_picked_index(1:end), :);

        %find the cost and gradient
        [t_cost_param, t_gradient_param] = function_Compute_Cost_Gradient...
            (...
            t_learnt_w,t_picked_X, t_picked_y, ...
            t_w2_filter_size, t_w2_bias_size,...
            t_w3_size, t_w4_size, ...
            t_n_conv_filter, t_reg_param...
            );
        
        %compute two main update parameter for adam
        t_adam_weight_movement = t_adam_param_beta1 .* t_adam_weight_movement + ( 1- t_adam_param_beta1) .* t_gradient_param;
        t_adam_weight_velocity = t_adam_param_beta2 .* t_adam_weight_velocity + ( 1 - t_adam_param_beta2) .* (t_gradient_param.^2);
        
        %correct the bias to boost up the parameter, this should not update
        %the original parameter
        t_adam_weight_movement_updater = t_adam_weight_movement ./ (1 - t_adam_param_beta1.^i);
        t_adam_weight_velocity_updater = t_adam_weight_velocity ./ ( 1- t_adam_param_beta2.^i);
        
        t_adam_updater = t_learning_rate .* t_adam_weight_movement_updater ./ (sqrt(t_adam_weight_velocity_updater) + t_adam_param_epsilon);
        
        %simply compute the gradient
        t_learnt_w = t_learnt_w - t_adam_updater;
        
        %decay the learning rate
        if (rem( i, t_learining_rate_decay_frequency) == 0)
            t_learning_rate = t_learning_rate * t_learning_rate_decay_ratio;
        end
            
        %output the cost to console every 100 iterate, so that we know
        %whether it is working, and the progress so far
        if( rem(i, t_record_frequency) == 0)
            %record the cost for plot
            t_record_cost_data(i/t_record_frequency) = t_cost_param;
            fprintf('update cost, current cost %.6f,\n',t_cost_param);
        end
        
        
    end
    
    r_learnt_weight = t_learnt_w;
    r_cost_history = t_record_cost_data;
    
    r_network_struct = struct(...
        't_n_input', t_n_input,...
        't_n_fc_hidden', t_n_fc_hidden,...
        't_n_conv_filter', t_n_conv_filter,...
        't_n_conv_filter_size', t_n_conv_filter_size,...
        't_n_conv', t_n_conv,...
        't_n_output', t_n_output,...
        't_w2_filter_size', t_w2_filter_size,...
        't_w2_bias_size', t_w2_bias_size,...
        't_w3_size', t_w3_size,...
        't_w4_size', t_w4_size...
        );
  