%This function will read the content from the data file
%output
%r_data_set: the whole data set
%r_pre_process_data_set: the data set has been pre-processsed
%r_answer_set: the whole answer set
function ...
    [...
    r_training_data_set,r_traning_answer_set,...
    r_test_dataset,r_test_answer_set,...
    r_data_label...
    ...
    ] = function_Read_Data_Files()    
    %Process the training data first
    %hard code the index here, the data file locates in Data directory 
    %put the prepared data into return blocks
    load('data_MNIST.mat');

    r_traning_answer_set = training_answer;
    r_training_data_set = training_data;
    r_data_label = ["1";"2";"3";"4";"5";"6";"7";"8";"9";"0"];
    
    %Process the test data    
    r_test_dataset = test_data;
    r_test_answer_set = test_answer;
    
end