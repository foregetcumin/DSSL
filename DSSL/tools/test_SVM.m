close all;
addpath libsvm-new;
% addpath data;
% addpath deepfeatures;
addpath HOG_fea;
%%
% src_str = {'CK','jaffe','CK','KDEF','KDEF','jaffe','CK','tfeid','jaffe','tfeid','KDEF','tfeid','FER','RAF'};
% tgt_str = {'jaffe','CK','KDEF','CK','jaffe','KDEF','tfeid','CK','tfeid','jaffe','tfeid','KDEF','RAF','FER'};
% src_str = {'CK','JAFFE'};
% tgt_str = {'JAFFE','CK'}
src_str = {'KDEF','TFEID','KDEF','CK','CK','FER'};
tgt_str = {'TFEID','JAFFE','JAFFE','KDEF','TFEID','RAF'};

fileID = fopen('sec_SVMHOG3.txt','a+');
all_acc=[];

for i = 1:6 % 14
    src = src_str{i};
    tgt = tgt_str{i};
    
    fprintf(fileID,'$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n %s\n',datestr(now));
%     fprintf(fileID,'第%d组  $$$$$$$$$$$$$$$ --%s vs %s-- $$$$$$$$$$$$$$\r\n',i,src,tgt);
%     fprintf('第%d组  $$$$$$$$$$$$$$$ --%s vs %s-- $$$$$$$$$$$$$$\n ',i, src, tgt);
    
%     load(['deepfeatures\' src '_rs.mat']);
    load(['HOG_fea\' src '_HOGwr.mat']);
    Xs = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2)); %数据的归一化操作
    Xs = zscore(Xs);%数据标准化：将数据变换为均值为0，标准差为1的分布切记，并非一定是正态的；
    Xs = normr(Xs)';%normr(X)接受单个矩阵或矩阵的单元数组，并返回行规范化为1的矩阵。
    Xs_label= trainIdx;%
    clear Train_DAT;
    clear trainIdx;
%     load(['deepfeatures\' tgt '_rs.mat']);
    load(['HOG_fea\' tgt '_HOGwr.mat']);
    %Xt = Train_DAT*diag(sparse(1./sqrt(sum(Train_DAT.^2))));%数据的归一化操作
    Xt = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));
    Xt=zscore(Xt);
    Xt = normr(Xt)';
    Xt_label= trainIdx;
    clear Train_DAT;
    clear trainIdx;
    
    tic;
    tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
    model = svmtrain(Xs_label, Xs', tmd);
    [~, acc] = svmpredict(Xt_label, Xt', model);
    acc = acc(1);
    time = toc;
    fprintf('第%d组  ******************************\n %s vs %s :\naccuracy: %.2f\ntime:%.3f\n',i,src,tgt,acc,time);
    fprintf(fileID,'第%d组  ******************************\n %s vs %s :\naccuracy:%.2f\ntime:%.3f\n ',i,src,tgt,time);
    fprintf(fileID,'%.2f\n', acc);
    all_acc=[all_acc;acc];
end

% fprintf('SVM accuracies in all groups:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',all_acc);
% fprintf(fileID,'SVM accuracies in all groups:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f \t \n',all_acc);
fprintf('SVM accuracies in all groups:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\t \n',all_acc);
fprintf(fileID,'SVM accuracies in all groups:\n %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\t \n',all_acc);
fclose(fileID);
fprintf(' the algorithm is over!\n');