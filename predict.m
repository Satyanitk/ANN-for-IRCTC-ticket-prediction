 
%Written by- 
%               Satyajit Gantayat,
%               V semester,BTech EEE, 
%               NITK Surathkal,Karnataka
%               Date-29-10-2018

clear
close all
clc

%% preparing data
%data=xlsread('dataset_train ticket.csv'); %reading the excel dataset file
load('data.mat'); %reading the data extracted from the excel file
[row,col]=size(data);
input=data(:,1:col-1); %extracting input and output from the dataset
output=data(:,col);

%% Dataset Visualisation
[in_row,in_col]=size(input);
t=[30,7,2,1];
pos_elem=zeros(row,4);
for i=1:row
    if input(i,5:8)>0==[1 1 1 1]
        pos_elem(i,:)=input(i,5:8);
    end
end
pos_elem= pos_elem(all(pos_elem,2),:);
tt=0:1.3:30;
for i=40:57
    plot(tt,pchip(t,pos_elem(i,:),tt));
    hold on;
end
title('Dataset Visualisation','color','b','FontSize',20);
xlabel('Time(in days)','color','r','FontSize',14);
ylabel('Ticket Status','color','r','FontSize',14);
grid on;

%% Defining Neural Network 
input=input';
targ=output';
net=feedforwardnet([20 20],'trainrp');
net.numInputs=3;
net.inputConnect=[1 1 1;0 0 0;0 0 0];

%input layers definition
net.inputs{1}.name='Travel class'; net.inputs{1}.size=1;
net.inputs{2}.name='Travel date,Month'; net.inputs{2}.size=2;
net.inputs{3}.name='Ticket Status'; net.inputs{3}.size=5;

%parameters modification
net.divideParam.trainRatio=0.8;
net.divideParam.valRatio=0.1;
net.divideParam.testRatio=0.1;

net.layers{1}.transferFcn='logsig';
net.layers{2}.transferFcn='logsig';

net.trainParam.mu=0.005;
net.trainParam.max_fail=8;

view(net)

%% training the network
[net,tr]=train(net,input,targ);
%Visualising the Performance
disp('performance of the network')
plotperform(tr)
%characteristics of the network
disp('characteristics of the network')
disp(tr)

%% Re-training of the obtained net for better performance if necessary
%net=init(net);
%[net,tr]=train(net,input,targ);
