function [f0,f1,g0,g1,d2f0,d2f1,d2g0,d2g1,h,hp,hpp,rhs,exact_sol] = setup()
f0 = @(y)y.^3;
f1 = @(y)(1+y.^3).*exp(-1);
g0 = @(x)x.*exp(-x);
g1 = @(x)(x+1).*exp(-x);
% differential operator is d^2/dx^2 + d^2/dy^2
% differential operator applied to A(x,y), the bdry term
d2f0 = @(y)6*y;
d2f1 = @(y)6*y.*exp(-1);
d2g0 = @(x)(x-2).*exp(-x);
d2g1 = @(x)(x-1).*exp(-x);
% differential operator applied to B(x,y) = x(1-x)y(1-y)NN(x,y,v,W,u)
h = @(x)x.*(1-x);
hp = @(x)1-2*x;
hpp = -2;
% right-hand side
rhs = @(x,y)exp(-x).*(x - 2 + y.^3 + 6*y);
exact_sol = @(x,y)exp(-x).*(x + y.^3);
end