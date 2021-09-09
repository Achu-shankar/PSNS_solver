close all
t = 0:0.00025:1.2;
plot(t(1:4000),Ens(1:4000),'b','linewidth',1.5)
hold on
plot(t_e,Ens_e,'sr','linewidth',2)
xlabel('time','fontsize',15)
ylabel('Enstropy','fontsize',15)
title("Enstropy Vs time")

ax = gca;
ax.FontSize = 16;
legend('Current solver','Reference result')
hold off

%%
plot(t(1:4000),KE(1:4000),'b','linewidth',1.5)
hold on
plot(t_e,KE_e,'sr','linewidth',2)
xlabel('time','fontsize',15)
ylabel('KE','fontsize',15)
title("KE Vs time")

ax = gca;
ax.FontSize = 16;
legend('Current solver','Reference result')