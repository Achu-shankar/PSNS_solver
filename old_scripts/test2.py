import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, sin, cos, exp
from scipy.fft import fft2,ifft2,fftfreq

nx=512
ny=512
L = 2*pi

dx = L/nx
dy = L/ny

x = np.linspace(0,L,nx)
y = np.linspace(0,L,ny)
X,Y = np.meshgrid(x,y,indexing='ij')

kx = fftfreq(nx,1/nx)*2*pi/L
kx[0] = 1.0e-6
ky = fftfreq(ny,1/ny)*2*pi/L
ky[0] = 1.0e-6



# print(kx.max()/(2.0*pi))

M,N = np.meshgrid(kx,ky,indexing='ij')

kmax_2b3_x = 0.5*nx/3*2*pi/L 
kmax_2b3_y = 0.5*ny/3*2*pi/L  
dealias = np.array(
    (np.abs(M)<kmax_2b3_x)*
    (np.abs(N)<kmax_2b3_y),
    dtype = bool
    )

# k2 = (np.square(M) + np.square(N))
# kx 
# U = np.ones(X.shape)
V = np.zeros(X.shape)
U = Y
V = -X

a_sq = 0.1**2
x1    =  X-0.5*L
y1    =  Y-0.5*L

r_sq1 = np.square(x1) + np.square(y1) + 1e-6

U  = -(y1*(1-np.exp(-r_sq1/a_sq))/r_sq1)
V  =  (x1*(1-np.exp(-r_sq1/a_sq))/r_sq1) 

Uf = fft2(U)
Vf = fft2(V)
Wf = (M*Vf-N*Uf)*1j
Wf_fil = Wf*dealias
# print(Wf.max())

W = np.real(ifft2(Wf))
W_fil = np.real(ifft2(Wf_fil))

W_fd = np.gradient(V,dx,axis=0) - np.gradient(U,dy,axis=1)

# print(W_fd.shape)

U = ifft2(Uf)
V = ifft2(Vf)

plt.contourf(X,Y,W_fd,levels=200,cmap='jet')
plt.colorbar()
plt.show()
# plt.plot(W[int(nx/2),:])
# plt.plot(W_fil[int(nx/2),:])
plt.plot(W_fd[int(nx/2),:])
plt.show()
W.sum()