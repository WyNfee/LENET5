%clear and close everything
clear; close all; clc;

%add searching path 
addpath(genpath('Data'))
addpath(genpath('Utils'))
addpath(genpath('Functions'))
addpath(genpath('Network'))

%read the CFIAR data
[...
    t_training_data, t_training_answer,...
    t_test_data, t_test_answer,...
    t_labels_string...
] =function_Read_Data_Files();

%preview the data
s = input('preview the image data?, y to yes:','s');
if(s == 'y')
    fprintf('Preview the image\n');   
    function_Preview_Image(t_training_data, t_training_answer, t_labels_string);
else
    fprintf('Skip image preview\n');
end

%Oraganize the training answer, since we have 10 classes
t_traning_answer_for_learning = function_Build_Answer_Matrix(t_training_answer,10);

%Machine Learning Algorithm
[t_learnt_weight, t_cost_history, t_network_struct] = function_Learning_Algorithm(t_training_data, t_traning_answer_for_learning);

%Check performance on training data
function_Check_Performance(t_training_data, t_training_answer, t_learnt_weight, t_network_struct);

%Check the performance on test data
function_Check_Performance(t_test_data, t_test_answer, t_learnt_weight, t_network_struct);


