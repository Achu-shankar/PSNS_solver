import numpy as np
from numpy import pi, cos, sin
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
# from scipy.fft import fftfreq, rfftn, irfftn
from numpy.fft import fftfreq , fft2 , ifft2 , \
irfft2 , rfft2 , irfftn , rfftn
import time

try:
  from pyfftw.interfaces.numpy_fft \
  import fft2, ifft2, irfft2, rfft2, irfftn , rfftn
  import pyfftw
  pyfftw.interfaces.cache.enable()
  # pyfftw.interfaces.cache.set_keepalive_time(60)
  pyfftw.config.NUM_THREADS = 6
  pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
  # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
except ImportError:
  print("Couldn't load pyfftw libraries, resorting back to numpy fft libraries")
  pass # Rely on numpy . fft routines

# dtype_re   = 'float32'
# dtype_comp = 'complex32'
# dtype_int  = 'int32'
dtype_re   = np.double
dtype_comp = np.cdouble
dtype_int  = np.int

class SolverVars:
  def __init__(self,xR=[0,1],yR=[0,1],nx=128,ny=128,nu=1e-4,dt=0.002,T=1,anim_s=20):

    self.xR = xR
    self.yR = yR
    self.nx = nx
    self.ny = ny
    self.Lx = xR[1]-xR[0]
    self.Ly = yR[1]-yR[0]
    self.dx = self.Lx/nx
    self.dy = self.Ly/ny
    self.Mesh  = np.empty((2,nx,ny),dtype = dtype_re)

    self.nu = nu

    self.dt = dt
    self.T  = T
    self.nt = int(T/dt)
    self.anim_s = anim_s
    self.it = 0

    nf  = ny
    self.Ut  = np.zeros((2,nx,ny,int(self.nt/anim_s+1)),dtype=dtype_re)
    self.U   = np.zeros((2,nx,ny),dtype=dtype_re)
    self.Uf  = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.Uf_ = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.Uf2_ = np.zeros((2,nx,nf),dtype=dtype_comp) # second intermediate value
    self.dUf = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.P   = np.zeros((nx,ny),dtype=dtype_re)
    self.Pf  = np.zeros((nx,nf),dtype=dtype_comp)
    self.W   = np.zeros((nx,nf),dtype=dtype_re) 
    self.Wf  = pyfftw.empty_aligned((nx,nf), dtype=dtype_comp)
    # self.Wf  = np.zeros((nx,nf),dtype=dtype_comp) 
    self.S   = np.zeros((nx,nf),dtype=dtype_re) 
    self.Sf  = np.zeros((nx,nf),dtype=dtype_comp) 

    self.Pdf   = np.zeros((2,nx,nf),dtype=dtype_comp)  # as this is actually (dp/dx, dp/dy) in fourier
    self.Visf  = np.zeros((2,nx,nf),dtype=dtype_comp)
    self.UxWf  = np.zeros((2,nx,nf),dtype=dtype_comp) #better than creating these at each and every time step
    # self.UxW   = np.zeros((2,nx,nf),dtype=dtype_re)
    self.UxW   = pyfftw.empty_aligned((2,nx,nf),dtype=dtype_re)

    self.K    = np.zeros((2,nx,ny))
    self.K_sq = np.zeros((nx,ny))
    self.dealias = np.zeros((nx,ny))
    self.K_K_sq =  np.zeros((2,nx,ny))

def solve_RK4(SV):  
  # RK-4 vars
  aa = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
  bb = np.array([0,0.5,0.5,1])
 
  for i in range(2): SV.Uf[i] = fft2(SV.U[i])
  for i in range(2): SV.Ut[i,:,:,0] = SV.U[i]

  # compute wavespace vectors
  wavespace(SV)

  for it in range(SV.nt+1):
    SV.Uf_ = np.copy(SV.Uf)
    for rk in range(4):
      if rk>0:
        SV.Uf2_ = SV.Uf_ + SV.dUf*bb[rk]*SV.dt
      else:
        SV.Uf2_ = np.copy(SV.Uf_)
      computeRHS(SV)
      SV.Uf = SV.Uf + aa[rk]*SV.dt*SV.dUf 

    if it%SV.anim_s ==0:
      print("Iteration == ", it)
      for i in range(2): SV.Ut[i,:,:,int(it/SV.anim_s)] = ifft2(SV.Uf[i])
    
    # CB_vars = callbacks(SV)

  for i in range(2): SV.Ut[i,:,:,int(it/SV.anim_s)] = ifft2(SV.Uf[i])
      
  # return SV,CB_vars

def wavespace(SV):
  kx = fftfreq(SV.nx,1/SV.nx)*2*pi/(SV.dx*SV.nx)
  kx[0] = 1.0e-6
  ky = fftfreq(SV.ny,1/SV.ny)*2*pi/(SV.dy*SV.ny)
  ky[0] = 1.0e-6

  SV.K[0],SV.K[1] = np.array(np.meshgrid(kx,ky,indexing='ij'))

  SV.K_sq   = np.sum(np.square(SV.K),axis=0)

  kmax_2b3_x = SV.nx/3*2*pi/(SV.dx*SV.nx) 
  kmax_2b3_y = SV.ny/3*2*pi/(SV.dy*SV.ny)  

  SV.dealias = np.array(
      (np.abs(SV.K[0])<kmax_2b3_x)*
      (np.abs(SV.K[1])<kmax_2b3_y),
      dtype = bool
      )
  SV.K_K_sq = SV.K.astype(dtype_re)/np.where(SV.K_sq==0,1,SV.K_sq).astype(dtype_re)

def generate_mesh(SV):
  x = np.arange(SV.xR[0],SV.xR[1],SV.dx,dtype = dtype_re)
  y = np.arange(SV.yR[0],SV.yR[1],SV.dy,dtype = dtype_re)
  SV.Mesh[0],SV.Mesh[1] = np.meshgrid(x,y,indexing='ij')

def computeRHS(SV):
  nf = SV.ny
  # non-linear term
  nonLin2D_2b3(SV) 

  # pressure term
  SV.Pdf = SV.K*np.sum(SV.K_K_sq*SV.UxWf,axis=0)

  # viscous term
  SV.Visf = SV.nu*SV.K_sq*SV.Uf2_

  # RHS
  SV.dUf = SV.UxWf - SV.Visf - SV.Pdf

def nonLin2D_2b3(SV):  
  Uf_ = SV.Uf2_*SV.dealias 
  for i in range(2): SV.U[i] = ifft2(Uf_[i])  
  SV.W = vorticity2D(Uf_,SV)
  
  SV.UxW[0]  =  SV.U[1]*SV.W
  SV.UxW[1]  = -SV.U[0]*SV.W
  SV.UxWf[0] =  fft2(SV.UxW[0])
  SV.UxWf[1] =  fft2(SV.UxW[1])
  SV.UxWf    =  SV.UxWf*SV.dealias 

def vorticity2D(Uf,SV): 
  SV.Wf = (1j*(SV.K[0]*Uf[1] - SV.K[1]*Uf[0]))
  return np.real(ifft2(SV.Wf))
  
def initial_condition(SV):
    # sL = 2*pi
    d_w = 80
    d_p = 0.05
    mask = SV.Mesh[1]<= 0.5
    n_mask = np.invert(mask)
    mask = mask.astype(float) 
    n_mask = n_mask.astype(float) 
    SV.U[0] = mask*np.tanh(d_w*(SV.Mesh[1]-0.25)) + n_mask*np.tanh(d_w*(0.75-SV.Mesh[1]))
    SV.U[1] = d_p*sin(2*pi*(SV.Mesh[0]))

t1 = time.time()
SV = SolverVars(dt=0.001)
generate_mesh(SV)
wavespace(SV)
initial_condition(SV)
solve_RK4(SV)
print(time.time()-t1)
  
Nt = -1
Ut = SV.Ut[:,:,:,Nt]
Uf = fft2(Ut)
W = np.zeros((SV.nx,SV.ny),dtype=dtype_re)
W = vorticity2D(Uf,SV)

plt.contourf(SV.Mesh[0],SV.Mesh[1],W,levels = 100, cmap='jet')
plt.show()