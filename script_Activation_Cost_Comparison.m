%clear Everything
clear; close all; clc;


%load the cost data
load('data_loss_tahn_softmax.mat')
load('data_loss_sigmoid_softmax.mat')

%plot them out
plot(t_cost_data_tanh(:,1), t_cost_data_tanh(:,2),'-', t_cost_data_sigmoid(:,1), t_cost_data_sigmoid(:,2), '--');
legend('Tanh','Sigmoid')
