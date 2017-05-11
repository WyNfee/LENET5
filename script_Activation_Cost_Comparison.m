%clear Everything
clear; close all; clc;


%load the cost data
load('data_loss_tanh_softmax.mat')
load('data_loss_sigmoid_softmax.mat')
load('data_loss_relu_softmax.mat');
load('data_loss_leaky_softmax.mat');
load('data_loss_elu_softmax.mat')

%plot them out
plot(t_cost_data_tanh(:,1), t_cost_data_tanh(:,2),'r-', ...
    t_cost_data_sigmoid(:,1), t_cost_data_sigmoid(:,2), 'm--',...
    t_cost_data_relu(:,1), t_cost_data_relu(:,2), 'b-.',...
    t_cost_data_leaky(:,1), t_cost_data_leaky(:,2), 'k--',...
    t_cost_data_elu(:,1), t_cost_data_elu(:,2), 'g-');
legend('Tanh','Sigmoid','ReLu','Leaky','ELu')
