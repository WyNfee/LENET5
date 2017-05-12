%clear Everything
clear; close all; clc;


%load the cost data
load('data_loss_relu_softmax_sgd.mat');
load('data_loss_sigmoid_softmax_sgd.mat');
load('data_loss_tanh_softmax_sgd.mat');

%plot them out
plot(t_cost_data_relu(:,1), t_cost_data_relu(:,2),'r-',...
        t_cost_data_sigmoid(:,1), t_cost_data_sigmoid(:, 2), 'g-',...
        t_cost_data_tanh(:,1), t_cost_data_tanh(:,2), 'b-'...
        );
legend('ReLu', 'Sigmoid','Tanh');
