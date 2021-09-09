clear all

 %Sim ul a ti o n P r ope r ty S e t ti n g
GridSize =128;
Visc = 0.001;

 % Space S e t ti n g
 h=2*pi / GridSize ;
 
 axis = h*[ 1 : 1 : GridSize ] ;
[ x , y]= meshgrid ( axis , axis ) ;
% Time S e t ti n g

FinTime=20;
dt = 0.01;
t =0;

% Movie F i l e Data A l l o c a t i o n Se t Up
FrameRate=100;
Mov(10)=struct( 'cdata' , [ ] , 'colormap' , [ ] ) ;

k=0;
j =1;

% D e fi ni n g I n i t i a l V o r t i c i t y D i s t r i b u t i o n
% exp(-())
% H = exp(-((x-pi+pi/5).^2+(y-pi+pi/5).^2 )/(0.3)) ...
% -exp (-((x-pi-pi/5 ).^2+(y-pi+pi/5 ) .^2 ) / ( 0.2 ) )+ ...
% exp (-((x-pi-pi/5 ).^2+(y-pi-pi/5).^2 )/(0.4 ) ) ;


xc1 = pi-pi/4;
yc1 = pi;
xc2 = pi+pi/4;
yc2 = pi;
H = exp(-pi*((x-xc1).^2 + (y-yc1).^2)) ...
  + exp(-pi*((x-xc2).^2 + (y-yc2).^2)) ;

% Adding Random N oi se t o I n i t i a l V o r t i c i t y
epsilon = 0.3;
Noise = random ( 'unif' , -1 ,1 , GridSize , GridSize );

% Note t h a t f o r Low V i s c o s i t i e s Adding N oi se t o Non−T r i v i a l V o r t i c i t y % D i s t r i b u t i o n r e s u l t s i n blow up , s o e i t h e r do pure
% n oi s e o r smooth data

% w = H+epsilon*Noise ;
w = H;

w_hat = fft2(w) ;

%%%%%%%%% Method Be gin s Here %%%%%%%%%%

kx=1i * ones( 1 , GridSize )'*(mod ( ( 1 : GridSize )-ceil( ...
GridSize/2+1) , GridSize )-floor( GridSize/ 2 ) ) ;

ky=1i*(mod ( ( 1 : GridSize )'- ceil( GridSize/2+1) , GridSize )- ...
floor( GridSize / 2 ) )*ones( 1 , GridSize ) ;

AliasCor = kx<2/3*GridSize & ky<2/3*GridSize ;
Lap_hat=kx.^2+ky.^2 ;

ksqr = Lap_hat ; ksqr(1,1) =1;

while t<FinTime

psi_hat = -w_hat./ ksqr ;
u =real( ifft2 ( ky .*psi_hat ) ) ;

v =real ( ifft2 (-kx.*psi_hat ) ) ;

w_x = real ( ifft2 ( kx.*w_hat ) ) ;

w_y = real ( ifft2 ( ky.*w_hat ) ) ;

VgradW = u.*w_x + v.*w_y ;
VgradW_hat = fft2(VgradW) ;

VgradW_hat = AliasCor.*VgradW_hat ;

%Crank−Ni c h ol s o n Update Method

w_hat_update =  1./( 1 / dt - 0.5*Visc*Lap_hat ).*( ( 1 / dt ...
+0.5* Visc*Lap_hat ).*w_hat-VgradW_hat) ;


if( k==FrameRate )

w = real (ifft2(w_hat_update ) ) ;

%Vel=s q r t (u.ˆ2+v . ˆ 2 ) ; %This i s f o r p l o t t i n g
% v e l o c i t y

contourf(x,y,w,80,'edgecolor','none') ;
colorbar ;

shading flat ; colormap ( 'jet' ) ;
drawnow
Mov( j )=getframe ;
k=0;
j=j+1;
end
w_hat=w_hat_update ;
t=t+dt ;
k=k+1;
end


 