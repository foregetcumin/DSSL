function [ACC,obj,pseudo_label,iter] = DSSL(X,Xs,Xt,c,Ys,Yt,max_iter_num,alpha,beta,lambda,gamma)

%%
m = size(X,1);%dim=m
n = size(X,2);
M = ones(n, c);
eta = 0.02;
P = ones(m,c);
H = ones(n,c);
Im = eye(m);
Ic = eye(c);
model=svmtrain(Ys,Xs','-s 0 -t 0 -c 1 -g 1 ');
[Ytrain_pseudo, ~, ~] = svmpredict(Yt,Xt',model);
Y=[Ys;Ytrain_pseudo];
F = Pre_label(Y);
B = 2 * F - ones(n, c);

options = [];
options.NeighborMode = 'KNN';
options.k = 10;
options.WeightMode = 'Binary';
S = constructW(X', options);
for j = 1:n
    d_sum = sum(S(j,:));
    if d_sum == 0
        d_sum = eps;
    end
    S(j,:) = S(j,:)/d_sum;
end
S = (S+S')/2;
D = diag(sum(S));
L = D - S;

Sy= constructW(F, options);
for j = 1:n
    d_sum = sum(Sy(j,:));
    if d_sum == 0
        d_sum = eps;
    end
    Sy(j,:) = Sy(j,:)/d_sum;
end
Sy = (Sy+Sy')/2;
Dy = diag(sum(Sy));
Ly = Dy - Sy;
%%

for iter = 1:max_iter_num

    %upadate P
    D_weight = diag( 0.5./sqrt(sum(P.*P,2)+eps));
    P = inv(2*X*X'+ 2*beta*D_weight+gamma*X*L*X'+gamma*X*L'*X'+2*alpha*Im)*2*(1+alpha)*X*H;
    
    %update H
    first = inv((1+lambda+eta)*Ic+alpha*P'*P);
    linshi_F = F + B.* M;
    H = ((1+alpha)*X'*P+lambda*linshi_F)*first;
    eta = eta*2;
    
    
    %update M
    M = max(B.* (H-F),0);
    
    
    %update Ytrain_pseudo
    Ht = Xt'*P;
    Ht = Ht*diag(sparse(1./sqrt(sum(Ht.^2))));
    [~,Ytrain_pseudo] = max(Ht',[],1);
    Y = [Ys;Ytrain_pseudo'];
    F = Pre_label(Y);
    
    
    %update B
    B = 2 * F - ones(n, c);
    
    %update L
    if iter == 1
            Weight = constructW_PKN(S, 15);
            Diag_tmp = diag(sum(Weight));
            L = Diag_tmp - Weight;
      else
            param.num_view = 15; 
            HG = gsp_nn_hypergraph(S, param);
            L = HG.L;
    end 
    
        %update Ly
    if iter == 1
            Weighty = constructW_PKN(Sy, 15);
            Diag_tmpy = diag(sum(Weighty));
            Ly = Diag_tmpy - Weighty;
      else
            param.num_view = 15; 
            HGy = gsp_nn_hypergraph(Sy, param);
            Ly = HGy.L;
    end
    
    norm21 = sum(sqrt(sum(P.^2,2)));
    %print OBJ
    obj(iter)=norm(X'*P-H,'fro')+alpha*norm(X-P*H','fro')+beta*norm21+lambda*norm(H-(F+B.*M),'fro')+gamma*(trace(P'*X*L*X'*P)+trace(F'*Ly*F));
    %收敛早停
    if (iter>1 &&(abs(double(obj(iter))-double(obj(iter-1)))< 10^-3))
        break;
    end
end

%%

num_train = size(Xs, 2); 
H = H';
H = H./repmat(sqrt(sum(H.^2)),size(H,1),1); 

if num_train < size(H,2)
    Hs = H(:, 1:num_train);  
    Ht = H(:, num_train+1:end); 
else
    error('索引超出范围: num_train 大于 H 的列数');
end

% SVM 训练
tmd = ['-s 0 -t 2 -g ' num2str(1e-3) ' -c ' num2str(1000)];
model = svmtrain(Ys, Hs', tmd);

% 预测
if isempty(model)
    error('SVM 训练失败，模型为空！');
end

[predict, ACC,~] = svmpredict(Yt, Ht', model);

pseudo_label = predict;

ACC = ACC(1);




