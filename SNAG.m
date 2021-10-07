function [fall,norg] = SNAG(nt,N,tol,iter_max)
fsz = 16; % fontsize
batch_size = 20;
%% setup training mesh
t = linspace(0,1,nt+2);
[xm,ym] = meshgrid(t,t);
I = 2:(nt+1);
xaux = xm(I,I);
yaux = ym(I,I);
xy = [xaux(:),yaux(:)]';
Ntrain = nt*nt; % the number of training points
%% initial guess for parameters
% N = 10; % the number of hidden nodes (neurons)
npar = 4*N; % the total number of parameters
wx = rand(npar,1); 
wy = wx;


%% The trust region BFGS method
tic % start measuring the CPU time
iter = 1;
norg = zeros(iter_max+1,0);
fall = zeros(iter_max+1,0);
nor = tol+1;
alpha = 0.005;
while nor > tol && iter < iter_max
    
    mu = 1 - 3/(5 + iter); 
    
    indices = randperm(Ntrain);
    indices = indices(1:batch_size);
    [r,J] = Res_and_Jac(wy,xy(:,indices));
    f = F(r);
    g = J'*r/Ntrain;
    nor = norm(g);

    wx_new = wy - alpha*g;
    
    wy_new = (1 + mu)*wx_new - mu*wx; 
    
    wx = wx_new;
    wy = wy_new;
    fprintf('iter %d: f = %d, norg = %d\n',iter,f,nor);
    norg(iter) = nor;
    fall(iter) = f;
    iter = iter + 1;
end
fprintf('iter # %d: f = %.14f, |df| = %.4e\n',iter,f,nor);
cputime = toc;
fprintf('CPUtime = %d, iter = %d\n',cputime,iter);
w = wx_new;
%% visualize the solution
nt = 101;
t = linspace(0,1,nt);
[xm,ym] = meshgrid(t,t);
[fun,~,~,~] = ActivationFun();
[v,W,u] = param(w);
[f0,f1,g0,g1,~,~,~,~,h,~,~,~,exact_sol] = setup();
A = @(x,y)(1-x).*f0(y) + x.*f1(y) + (1-y).*(g0(x)-((1-x)*f0(0)+x*f1(0))) + ...
     y.*(g1(x)-((1-x)*f0(1)+x*f1(1)));
B = h(xm).*h(ym);
NNfun = zeros(nt);
for i = 1 : nt
    for j = 1 : nt
        x = [xm(i,j);ym(i,j)];
        NNfun(i,j) = v'*fun(W*x + u);
    end
end
sol = A(xm,ym) + B.*NNfun;
esol = exact_sol(xm,ym);
err = sol - esol;
fprintf('max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));
fprintf(fopen('SNAG_metrics.text', 'w'), 'max|err| = %d, L2 err = %d\n',max(max(abs(err))),norm(err(:)));

%
figure(1);clf;
contourf(t,t,sol,linspace(min(min(sol)),max(max(sol)),20));
colorbar;
set(gca,'Fontsize',fsz);
title('Computed solution from stochastic NAG');
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);
saveas(gcf,'stochastic_NAG_solution.png');

%
figure(2);clf;
contourf(t,t,err,linspace(min(min(err)),max(max(err)),20));
colorbar;
set(gca,'Fontsize',fsz);
title('Error of stochastic NAG w.r.t exact solution');
xlabel('x','Fontsize',fsz);
ylabel('y','Fontsize',fsz);
saveas(gcf,'stochastic_NAG_error.png');

%
figure(3);clf;
subplot(2,1,1);
fall(iter+2:end) = [];

plot((1:iter-1)',fall,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('f','Fontsize',fsz);
subplot(2,1,2);
norg(iter+2:end) = [];
plot((1:iter-1)',norg,'Linewidth',2,'Marker','.','Markersize',20);
grid;
set(gca,'YScale','log','Fontsize',fsz);
xlabel('k','Fontsize',fsz);
ylabel('|| grad f||','Fontsize',fsz);
saveas(gcf,'stochastic_NAG_convergence.png');
end

%% the objective function
function f = F(r)
    f = 0.5*r'*r/length(r);
end
