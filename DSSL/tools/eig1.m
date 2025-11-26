function [eigvec, eigval, eigval_full] = eig1(A, c, isMax, isSym)
% A:laplacian matrix;c:cluster number;isMax:0
% isMax与论文中求解最大的还是最小的特征值对应的特征向量有关。

if nargin < 2 % nargin 针对当前正在执行的函数，返回函数调用中给定函数输入参数的数目。
    % 该语法仅可在函数体内使用。使用 arguments 验证代码块时，函数内 nargin 返回的值是调用函数时提供的位置参数的个数。
    c = size(A,1);
    isMax = 1;
    isSym = 1;
elseif c > size(A,1)
    c = size(A,1);
end; % 多分支if语句

if nargin < 3
    isMax = 1;
    isSym = 1;
end;

if nargin < 4
    isSym = 1;
end;

if isSym == 1
    A = max(A,A');
end;
[v d] = eig(A);%v:eigen vector(column);d=eigen value(diagonal matrix)
% eigs——特征值和特征向量的子集；eig——特征值和特征向量。
d = diag(d);
%d = real(d);
if isMax == 0
    [d1, idx] = sort(d); % 如果 A 是向量，则 sort(A) 对向量元素进行排序。默认为升序。
else
    [d1, idx] = sort(d,'descend');
end;

idx1 = idx(1:c);
eigval = d(idx1);
eigvec = v(:,idx1);

eigval_full = d(idx); % 将排序后的全部特征值返回