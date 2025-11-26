clc; 
clear all;
close all;
addpath LapLRR
load('CK_LBP2.mat')
Xs=Train_DAT';
XXs=Train_DAT;
load('jaffe_LBP2.mat')
Xt=Train_DAT';
XXt=Train_DAT;
X=[XXs;XXt];
X=X';
%--------------------------------------------------------------------------
maxiter = 1e6;
mu = 0.5;
%--------------------------------------------------------------------------
lambda_lr = 0.12;
lambda_graph = 0.01;
%--------------------------------------------------------------------------
[Zest_lap,Eest_lap] = LapLRR(X,Xs,'LAMBDA_LR',lambda_lr,'LAMBDA_GRAPH',lambda_graph,'AL_ITERS',maxiter,'POSITIVITY','yes','MU',mu,'VERBOSE','yes','TOL',1e-8);
% -------------------------------------------------------------------------


%
















