function init_weight = function_Utils_GenericWeigthInitialization(p_input_size, p_output_size)
    epsilon_init = 0.12;
    init_weight = rand(p_output_size, p_input_size) * 2 * epsilon_init - epsilon_init;
end