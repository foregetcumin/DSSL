srcStr = {'CK','jaffe','CK','KDEF','KDEF','jaffe','CK','tfeid','jaffe','tfeid','KDEF','tfeid'}; % ,'FER','RAF'
tgtStr = {'jaffe','CK','KDEF','CK','jaffe','KDEF','tfeid','CK','tfeid','jaffe','tfeid','KDEF'}; % ,'RAF','FER'

for iData = 1:12
                
    src = char(srcStr{iData});
    tgt = char(tgtStr{iData});
%     g_title = strcat(src,'——',tgt);
%     
%     
%     %% AN 1; DI 2; FE 3; HA 4; SA 5; SU 6.
%     load(['./data/' src '_LBP.mat']);  % load('.\data\TFEID_LBP.mat');
%     Xs = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2)); %数据的归一化操作
%     Xs = zscore(Xs);%数据标准化：将数据变换为均值为0，标准差为1的分布切记，并非一定是正态的；
%     Xs = normr(Xs);%normr(X)接受单个矩阵或矩阵的单元数组，并返回行规范化为1的矩阵。
%     clear Train_DAT;
%     Ys = trainIdx;
%     clear trainIdx;
%     load(['./data/' tgt '_LBP.mat']);  % load('.\data\KDEF_LBP.mat');
%     Xt = Train_DAT ./ repmat(sum(Train_DAT,2),1,size(Train_DAT,2));
%     Xt=zscore(Xt);
%     Xt = normr(Xt);
%     clear Train_DAT;
%     Yt = trainIdx;
%     clear trainIdx;
%     
%     X = [Xs',Xt']; % m*n
%     % 由于tsne标准输入数据以行向量表示，因此先转置.
%     Y = tsne(X','Algorithm','exact','Distance','correlation','NumDimensions',2);% Chebychev,cityblock,cosine,euclidean,spearman,minkowski,correlation,mahalanobis. 'NumPCAComponents',6,
%     % Y = tsne(X');
%     Y1=Y(1:size(Xs,1),:);
%     Y2=Y(size(Xs,1)+1:end,:);
%     figure;
%     scatter(Y1(Ys==1,1),Y1(Ys==1,2),'*','r','LineWidth',1);
%     hold on
%     scatter(Y1(Ys==2,1),Y1(Ys==2,2),'*','b','LineWidth',1);
%     hold on;
%     scatter(Y1(Ys==3,1),Y1(Ys==3,2),'*','g','LineWidth',1);
%     hold on;
%     scatter(Y1(Ys==4,1),Y1(Ys==4,2),'*','y','LineWidth',1);
%     hold on;
%     scatter(Y1(Ys==5,1),Y1(Ys==5,2),'*','k','LineWidth',1);
%     hold on;
%     scatter(Y1(Ys==6,1),Y1(Ys==6,2),'*','m','LineWidth',1);
%     hold on;
%     
%     scatter(Y2(Yt==1,1),Y2(Yt==1,2),'+','r','LineWidth',1);
%     hold on
%     scatter(Y2(Yt==2,1),Y2(Yt==2,2),'+','b','LineWidth',1);
%     hold on;
%     scatter(Y2(Yt==3,1),Y2(Yt==3,2),'+','g','LineWidth',1);
%     hold on;
%     scatter(Y2(Yt==4,1),Y2(Yt==4,2),'+','y','LineWidth',1);
%     hold on;
%     scatter(Y2(Yt==5,1),Y2(Yt==5,2),'+','k','LineWidth',1);
%     hold on;
%     scatter(Y2(Yt==6,1),Y2(Yt==6,2),'+','m','LineWidth',1);
%     hold on;
% %     title(g_title);
%     xlabel(g_title);
%     % box on;
%     set( gca, 'FontSize',15,'FontWeight','bold' );
% set(gca,'fontsize',17,'fontname','Times','FontWeight','bold');
    


%% after: AN 1; DI 2; FE 3; HA 4; SA 5; SU 6.
    srcA = {'C','J','C','K','K','J','C','T','J','T','K','T'}; % ,'F','R'
    tgtA = {'J','C','K','C','J','K','T','C','T','J','T','K'}; % ,'R','F'
    src2 = char(srcA{iData});
    tgt2 = char(tgtA{iData});
    g_title2 = strcat(src2,'——',tgt2);
    
    load(['.\data\CK_LBP.mat']); % load('.\data\TFEID_LBP.mat');
    Xs = Train_DAT;
    Ys = trainIdx;
    clear trainIdx;
    load(['.\data\KDEF_LBP.mat']);  % load('.\data\KDEF_LBP.mat');
    Yt = trainIdx;
    clear trainIdx;
    
    combine = strcat(src2,tgt2);
    load(['./data_visualization/matrixH/' combine '_H.mat']);  % load('.\data_visualization\TK_H.mat'); % H d*n
    H = H'; % H n*d
%     H = H ./ repmat(sum(H,2),1,size(H,2)); %数据的归一化操作
%     H = zscore(H);  % 数据标准化：将数据变换为均值为0，标准差为1的分布切记，并非一定是正态的；
%     H = normr(H);  % normr(X)接受单个矩阵或矩阵的单元数组，并返回行规范化为1的矩阵。
    
    % 由于tsne标准输入数据以行向量表示，因此先转置.
    YH = tsne(H,'Algorithm','exact','Distance','euclidean', 'NumDimensions',2);% mahalanobis,Chebychev,cityblock,cosine,euclidean,spearman,minkowski,correlation. 'NumPCAComponents',6,
    % YH = tsne(H);
    Y3=YH(1:size(Xs,1),:);
    Y4=YH(size(Xs,1)+1:end,:);
    figure;
    
    scatter(Y3(Ys==1,1),Y3(Ys==1,2),'*','r','LineWidth',1);
    hold on
    scatter(Y3(Ys==2,1),Y3(Ys==2,2),'*','b','LineWidth',1);
    hold on;
    scatter(Y3(Ys==3,1),Y3(Ys==3,2),'*','g','LineWidth',1);
    hold on;
    scatter(Y3(Ys==4,1),Y3(Ys==4,2),'*','y','LineWidth',1);
    hold on;
    scatter(Y3(Ys==5,1),Y3(Ys==5,2),'*','k','LineWidth',1);
    hold on;
    scatter(Y3(Ys==6,1),Y3(Ys==6,2),'*','m','LineWidth',1);
    hold on;
    
    scatter(Y4(Yt==1,1),Y4(Yt==1,2),'+','r','LineWidth',1);
    hold on
    scatter(Y4(Yt==2,1),Y4(Yt==2,2),'+','b','LineWidth',1);
    hold on;
    scatter(Y4(Yt==3,1),Y4(Yt==3,2),'+','g','LineWidth',1);
    hold on;
    scatter(Y4(Yt==4,1),Y4(Yt==4,2),'+','y','LineWidth',1);
    hold on;
    scatter(Y4(Yt==5,1),Y4(Yt==5,2),'+','k','LineWidth',1);
    hold on;
    scatter(Y4(Yt==6,1),Y4(Yt==6,2),'+','m','LineWidth',1);
    hold on;
    % box on;
    title(g_title2);
    % set( gca, 'FontSize',15,'FontWeight','bold' );
    set(gca,'fontsize',17,'fontname','Times','FontWeight','bold');
    % set( gca, 'XTick', [], 'YTick', [],'ZTick', [],'FontSize',17,'FontWeight','bold' );
end