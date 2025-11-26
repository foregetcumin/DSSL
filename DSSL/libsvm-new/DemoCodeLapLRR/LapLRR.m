function [Z,E] = LapLRR(A,X,varargin)
% Laplacian regularized Low-Rank Representation (LapLRR)
%--------------------------------------------------------------------------
% min ||X-AZ||_F^2 + lambda_lrr *||Z||_*  + lambda_graph Tr(Z*L*Z^T)
%                    Z>=0;
%--------------------------------------------------------------------------
%Tuned parameters include:
% the maximum iteration: AL_iter
% the low-rank regularization parameter: lambda_lr
% the graph regularization parameter: lambda_graph
% the penalty parameter: mu
% to construct the graph, you need to tune the following parameters:
% the NeighborMode, NeighborNumber--k, WeightMode,
% preset the stoping tolerance: tol
% additionally, some third-party functions (constructW, EuDist2, and NormalizeFea) written by Deng Cai
% available at: http://www.cad.zju.edu.cn/home/dengcai/Data/data.html
% are needed before running the LapLRR.
[mA,c] = size(A);
[mX,N] = size(X);
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
reg_lr    = 0;
reg_graph = 0;
reg_pos   = 0;

AL_iters  = 1e6;
mu = 0.5; 
tol = 1e-3;

verbose = 'off';
Z0 = 0;
true_z = 0;

%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_LR'
                lambda_lr = varargin{i+1};
                if lambda_lr < 0
                    error('lambda must be positive');
                elseif lambda_lr > 0
                    reg_lr = 1;
                end
            case 'LAMBDA_GRAPH'
                lambda_graph = varargin{i+1};
                if lambda_graph < 0
                    error('lambda must be positive');
                elseif lambda_graph > 0
                    reg_graph = 1;
                end
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'POSITIVITY'
                positivity = varargin{i+1};
                if strcmp(positivity,'yes')
                    reg_pos = 1;
                end
            case 'MU'
                mu = varargin{i+1};
                if mu <= 0
                    error('mu must be positive');
                end
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'TOL'
                tol = varargin{i+1};
            case 'Z0'
                Z0 = varargin{i+1};
            case 'TRUE_Z'
                ZT = varargin{i+1};
                true_z = 1;
            otherwise
                error(['Unrecognized option: ''' varargin{i} '''']);
        end
    end
end
%---------------------------------------------
%  Constants and initializations
%---------------------------------------------
% number of regularizers
n_reg =  reg_lr + reg_pos + reg_graph;
IF = inv(A'*A + n_reg*eye(c));
%---------------------------------------------
%  Initializations
%---------------------------------------------
if Z0 == 0
    Z = IF*A'*X;
end

% initialize V variables
V = cell(1 + n_reg,1);
D = cell(1 + n_reg,1);

index = 1;
reg(1) = 1;                  % regularizers
V{index} = A*Z;              % V1
D{index} = zeros(size(X));   % Lagrange multipliers
index = index + 1;

if reg_pos == 1
    reg(index) = 2;
    V{index} = Z;
    D{index} = zeros(size(Z));
    index = index +1;
end

if reg_lr == 1
    reg(index) = 3;
    V{index} = Z;
    D{index} = zeros(size(Z));
    index = index +1;
end

if reg_graph == 1
    reg(index) = 4;
    V{index} = Z;
    D{index} = zeros(size(Z));
    
    %----------------------------------------------------------------------
    options = [];
    options.NeighborMode = 'KNN';
    options.k = 5;
    options.WeightMode = 'Cosine';
    options.t = 1;
    W_graph = constructW(X',options); 
    D_graph = full(sum(W_graph,2));
    D_graph = spdiags(D_graph,0,N,N);
    L_graph = D_graph - W_graph;
end
%---------------------------------------------
%  AL iterations - main body
%---------------------------------------------

i=1;
stopC = inf;

%--------------------------------------------------------------------------
while (i <= AL_iters) && (stopC >= tol)

    if mod(i,10) == 1
        Z0 = Z;
    end
    % ---------------------------------------------------------------------
    % Step 1: Update U
    Ztemp = A'*(V{1}+D{1});
    for j = 2:(n_reg+1)
        Ztemp = Ztemp+ V{j} + D{j};
    end
    Z = IF*Ztemp;
    
    %----------------------------------------------------------------------
    % Step 2: Update V1 for date fitting; V2 for positivity; V3 for addone;
    % V4 for L12 norm; V5 for graph regularization;
    for j=1:(n_reg+1)
        % Update V1
        if  reg(j) == 1
            V{j} = (1/(1+mu)*(X+mu*(A*Z-D{j})));
        end
        % Update V2
        if  reg(j) == 2
            V{j} = max(Z-D{j},0);
        end
        % Update V4
        if  reg(j) == 3
            V{j} = SVT(Z-D{j},lambda_lr/mu);
        end
        % Update V5
        if  reg(j) == 4
            V{j} = (Z - D{j})/(lambda_graph/mu*L_graph+eye(N));
        end
        
    end
    
    %----------------------------------------------------------------------
    % update Lagrange multipliers
    for j=1:(n_reg+1)
        if  reg(j) == 1
            mse = A*Z-V{j};
            D{j} = D{j} - mse;
            stopC = norm(mse,'fro');
        else
            mse = Z-V{j};
            D{j} = D{j} - mse;
            stopC = max(stopC,norm(mse,'fro'));
        end
    end
    
    % to make the primal residual and the dual residual keep closer
    if mod(i,10) == 1
        % primal residue
        res_p = stopC;
        % dual residue
        res_d = norm(Z-Z0,'fro');
        % update mu
        if res_p > 10*res_d
            mu = mu*2;
            for j=1:(n_reg+1)
               D{j}=D{j}/2; 
            end
        elseif res_d > 10*res_p
            mu = mu/2;
            for j=1:(n_reg+1)
               D{j}=D{j}*2; 
            end
        end
    end
    
    if  strcmp(verbose,'yes') 
        if (i ==1 || mod(i,10)==0 || stopC<tol)
        fprintf(strcat(sprintf('iter = %2i - Stop Criterion = %2.8f\t',i, stopC),'\n'));
        end
    end
    
    % compute RMSE
    if true_z
        rmse = norm(Z-ZT,'fro');
        if  strcmp(verbose,'yes')
            fprintf(strcat(sprintf('\t iter = %i - ||Z_est - Z||_F = %2.3f,  Rank(Z) = %d',i, rmse,rank(Z,1e-3*norm(Z,2))),'\n'));
        end
    end
    
    i=i+1;
end

E = X - A*Z;

function X = SVT(W,epsilon)
%Singular Value Thresholding operator for minimizing
%    \min \epsilon||X||_* + 1/2 ||X-W||_F^2
%
[U,S,V] = svd(W,'econ');
s = max(diag(S)-epsilon, 0) + min(diag(S)+epsilon,0);
X = U*diag(s)*V';









