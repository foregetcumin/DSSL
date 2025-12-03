addpath libsvm-new;
addpath data;
addpath tools;


%% load data 
load("KDEF_LBP.mat");
Train_DAT = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));  
Xs = zscore(Train_DAT); 
Xs = normr(Xs)';


clear Train_DAT;  
Ys = trainIdx; clear trainIdx;
%______________________________________________________________________________________________________________
load("CK_LBP.mat"); 
Train_DAT = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));  
Xt = zscore(Train_DAT); 
Xt = normr(Xt)';
c = length(unique(Ys));

clear Train_DAT;  
Yt = trainIdx; clear trainIdx;

%% set paras
rng(2025);
X = [Xs,Xt];
max_iter_num = 60;

alpha = 0.01;
beta = 1;
lambda = 0.1;
gamma = 10;
[ACC,obj,pseudo_label,iter] = DSSL(X,Xs,Xt,c,Ys,Yt,max_iter_num,alpha,beta,lambda,gamma);
disp(ACC);
disp(iter);
fprintf('The algorithm is over!\n');


