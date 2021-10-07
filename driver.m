function driver()
close all
fsz = 16; % Fontsize
nt = 5; % trial mesh is nt-by-nt
N = 10; % the number of neurons
tol = 1e-4; % stop if ||J^\top r|| <= tol
iter_max = 10000;  % max number of iterations allowed
%[~,~] = GD(nt,N,tol,iter_max);
%[~,~] = SG(nt,N,tol,iter_max);
%[~,~] = NAG(nt,N,tol,iter_max);
%[~,~] = SNAG(nt,N,tol,iter_max);
%[~,~] = Adam(nt,N,tol,iter_max);
[~,~] = SAdam(nt,N,tol,iter_max);

%
% figure(3);clf;
% subplot(2,1,1);
% hold on;
% plot((1:length(GDf))',GDf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','GD');
% plot((1:length(SGf))',SGf,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SG');
% legend;
% grid;
% set(gca,'YScale','log','Fontsize',fsz);
% xlabel('k','Fontsize',fsz);
% ylabel('f','Fontsize',fsz);
% subplot(2,1,2);
% hold on;
% plot((1:length(GDg))',GDg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','GD');
% plot((1:length(SGg))',SGg,'Linewidth',2,'Marker','.','Markersize',20,'Displayname','SG');
% legend
% grid;
% set(gca,'YScale','log','Fontsize',fsz);
% xlabel('k','Fontsize',fsz);
% ylabel('|| grad f||','Fontsize',fsz);
% saveas(gcf,'convergence_plots.png');

end