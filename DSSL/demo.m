addpath libsvm-new;
addpath data;
addpath tools;
warning off;
setenv('BLAS_VERSION','')


%% load data 
load("KDEF_LBP.mat");% 源域
Train_DAT = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));  % 行归一化
Xs = zscore(Train_DAT); % z-score标准化
Xs = normr(Xs)';%接受单个矩阵或矩阵的单元数组，并返回行规范化为1的矩阵


clear Train_DAT;  
Ys = trainIdx; clear trainIdx;
%______________________________________________________________________________________________________________
load("CK_LBP.mat");  % 目标域
Train_DAT = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));  % Train_DAT是这个mat文件的变量，大小为N*M
Xt = zscore(Train_DAT); 
Xt = normr(Xt)';
c = length(unique(Ys));%类别的数量

clear Train_DAT;  
Yt = trainIdx; clear trainIdx;

%% set paras
X = [Xs,Xt];
max_iter_num = 60;
rng(2025);
alpha = 0.01;
beta = 1;
lambda = 0.1;
gamma = 10;
[ACC,obj,pseudo_label,iter] = DSSL(X,Xs,Xt,c,Ys,Yt,max_iter_num,alpha,beta,lambda,gamma);
disp(ACC);
disp(iter);
fprintf('The algorithm is over!\n');

