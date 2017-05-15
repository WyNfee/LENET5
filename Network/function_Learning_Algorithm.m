%the core learning algorithm locate
%param:
%p_training_data: the training data set
%p_training_answer: the training answer
%output
%r_learnt_weight: the weight learnt in this training data set
%r_cost_history: the cost history used to plot learning performance
%r_network_struct: the information about this learning algorithm
function[r_learnt_weight, r_cost_history, r_network_struct] = function_Learning_Algorithm(p_training_data, p_training_answer)
    %set up architecture parameters
    t_layer_input_amount = 400;
    t_layer_hidden_amount = 100;
    t_layer_output_amount = 10;
    
    %reorganize the answer data
    t_training_data_amount = size(p_training_answer, 1);
    
    %init the weight for layer one
    t_layer_input_input_size = t_layer_input_amount + 1;
    t_layer_input_weight = function_XavierInitialization_For_ReLu(t_layer_input_input_size, t_layer_hidden_amount);

    %init the weight for layer two
    t_layer_hidden_input_size = t_layer_hidden_amount + 1;
    t_layer_hidden_weight = function_XavierInitialization_For_ReLu(t_layer_hidden_input_size, t_layer_output_amount);

    %pack the weigth together to compute the weight
    t_packed_weight = [t_layer_input_weight(:); t_layer_hidden_weight(:)]; 

    %provide the layer one and layer two size
    t_layer_input_weight_size = size(t_layer_input_weight);
    t_layer_hidden_weight_size = size(t_layer_hidden_weight);

    %a hyper parameter of regularization param, close the regularization here
    t_reularization_param = 0.07;

    %assign the hyperparameter learning rate
    t_learning_rate = 0.01;

    %assgin the hyperparameter, stochasic data size
    t_stochasitic_data_size = 50;

    %a storage for weigth in sgd 
    t_packed_weight_sgd  = t_packed_weight;

    %define the iteration time, 1000 * 100 times
    t_iteration_time = 10000;

    %record frequency
    t_record_frequency = 100;

    %a cost data storage for further ploting
    t_record_cost_data = zeros(t_iteration_time/t_record_frequency, 1);

    %using momentum optimizer
    t_momentum_param = 0.9;
    t_momentum_updater = zeros(size(t_packed_weight_sgd));

    %do gradient descent
    for i = 1: t_iteration_time
        
        %this will make the data always within [1:g_input_answer_amount]
        t_rand_picked_data_index = floor(rand(1, t_stochasitic_data_size) * (t_training_data_amount - 1) ) + 1;
        %generate stocastic data from dataset
        t_rand_picked_data = p_training_data(t_rand_picked_data_index(1:end), :);
        %provide the coresponding answer data as well
        t_rand_picked_answer = p_training_answer(t_rand_picked_data_index(1:end), :);

        %find the cost and gradient
        [t_cost_param, t_gradient_param] = function_Compute_Cost_Gradient(t_packed_weight_sgd,t_rand_picked_data, t_rand_picked_answer, t_layer_input_weight_size, t_layer_hidden_weight_size, t_reularization_param);
        
       %compute two main update parameter for adam
        t_momentum_updater = t_momentum_param * t_momentum_updater + t_learning_rate * t_gradient_param;
        t_packed_weight_sgd = t_packed_weight_sgd - t_momentum_updater;
        
        %output the cost to console every 100 iterate, so that we know
        %whether it is working, and the progress so far
        if( rem(i, t_record_frequency) == 0)
            %record the cost for plot
            t_record_cost_data(i/t_record_frequency) = t_cost_param;
            fprintf('update cost, current cost %.6f,\n',t_cost_param);
        end
        
        
    end
    
    r_learnt_weight = t_packed_weight_sgd;
    r_cost_history = t_record_cost_data;
    
    r_network_struct = struct(...
        't_layer_input_amount', t_layer_input_amount,...
        't_layer_hidden_amount', t_layer_hidden_amount,...
        't_layer_output_amount', t_layer_output_amount,...
        't_layer_input_weight_size', t_layer_input_weight_size,...
        't_layer_hidden_weight_size', t_layer_hidden_weight_size...
        );
  