clc; 
clear all;
close all;
%--------------------------------------------------------------------------
nClass = 5;                 % number of subspaces
nSamples = 40;              % number of sampled points in each subspaces
nTotal = nClass*nSamples;   % total number of sampled points
dAmbient = 200;             % dimension of ambient space
dManifold = 4;              % dimension of manifold mbedded in ambiend space
labels = ones(nTotal,1);
spaIndex = randperm(dAmbient);
A = orth(rand(dAmbient));
A = A(:,spaIndex(1:nClass*dManifold));

Z1 = rand(dManifold,nSamples);
for i=2:nClass
    Z2 = rand(dManifold,nSamples);
    Z = [Z1 zeros(size(Z1,1),size(Z2,2));
        zeros(size(Z2,1),size(Z1,2)) Z2];
    labels((i-1)*nSamples+1:i*nSamples) = i*ones(nSamples,1);
    Z1 = Z;
end

X = A*Z;

norm_x = sqrt(sum(X.^2,1));
norm_x = repmat(norm_x,dAmbient,1);

i = 1;
nX = size(X,2);

gn = norm_x.*randn(dAmbient,nX);
inds = rand(1,nX)<=(i-1)*5/100;
X_noise = X;
X_noise(:,inds) = X(:,inds) + 0.3*gn(:,inds);

%--------------------------------------------------------------------------
maxiter = 1e6;
mu = 0.5;
%--------------------------------------------------------------------------
lambda_lr = 0.12;
lambda_graph = 0.01;
%--------------------------------------------------------------------------
[Zest_lap,Eest_lap] = LapLRR(A,X_noise,'LAMBDA_LR',lambda_lr,'LAMBDA_GRAPH',lambda_graph,'AL_ITERS',maxiter,'POSITIVITY','yes','MU',mu,'VERBOSE','yes','TOL',1e-8);
% -------------------------------------------------------------------------

figure(1)
subplot(2,1,1)
imagesc(Z);   axis equal;axis tight; axis off;
title('True representation matrix');
subplot(2,1,2)
imagesc(Zest_lap);axis equal;axis tight; axis off; 
title('Estimated representation matrix by LapLRR')




%
















