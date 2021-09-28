function [v,W,u] = param(par)
N = length(par)/4;
v = par(1 : N);
W = reshape(par(N+1 : 3*N),[N,2]);
u = par(3*N+1 : end);
end