clc; clear; close all;

addpath(genpath('Data'))

load('data_MNIST_old.mat');

randIndex = randperm(5000,5000);


training_data = X(randIndex(1:4000),:);
training_answer = y(randIndex(1:4000),:);
test_data = X(randIndex(4001:5000), :);
test_answer = y(randIndex(4001:5000), :);

save('datafile.mat','training_data','training_answer','test_data','test_answer');
