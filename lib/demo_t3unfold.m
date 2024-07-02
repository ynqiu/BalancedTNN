%
clc;
clear;
A = rand(2,3,4,5,6,7);
xSize=size(A);
N = ndims(A);
for i=1:N
    X3 = t3unfold(A,i);
    size(X3)
    Xk = t3fold(X3,xSize,i);
    size(Xk)
end