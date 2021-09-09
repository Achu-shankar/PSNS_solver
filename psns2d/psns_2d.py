import numpy as np
from numpy import pi as PI
import pickle
from scipy.fft import fftfreq, irfft2, rfft2 
try:
  from pyfftw.interfaces.scipy_fft import irfft2, rfft2
  import pyfftw
  pyfftw.interfaces.cache.enable()
  pyfftw.interfaces.cache.set_keepalive_time(60)
  pyfftw.config.NUM_THREADS = 4
  pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
  # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
except ImportError:
  print("Couldn't load pyfftw libraries, resorting back to scipy fft libraries")
  pass # Rely on scipy . fft routines

from data_types import *


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
    self.Mesh  = np.empty((2,nx,ny),dtype = DTYPE_RE)

    self.nu = nu

    self.dt = dt
    self.T  = T
    self.nt = int(T/dt)
    self.anim_s = anim_s
    self.it = 1

    # nf  = ny
    nf = int(ny/2+1)
    self.Ut  = np.zeros((2,nx,ny,int(self.nt/anim_s+1)),dtype=DTYPE_RE)
    self.U   = np.zeros((2,nx,ny),dtype=DTYPE_RE)
    self.Uf  = np.zeros((2,nx,nf),dtype=DTYPE_COMP) 
    self.Uf_ = np.zeros((2,nx,nf),dtype=DTYPE_COMP) 
    self.Uf2_ = np.zeros((2,nx,nf),dtype=DTYPE_COMP) # second intermediate value
    self.Uf_temp = np.zeros((2,nx,nf),dtype=DTYPE_COMP)
    self.dUf = np.zeros((2,nx,nf),dtype=DTYPE_COMP) 
    self.P   = np.zeros((nx,ny),dtype=DTYPE_RE)
    self.Pf  = np.zeros((nx,nf),dtype=DTYPE_COMP)
    self.W   = np.zeros((nx,ny),dtype=DTYPE_RE) 
    self.Wf  = np.zeros((nx,nf),dtype=DTYPE_COMP) 
    self.S   = np.zeros((nx,nf),dtype=DTYPE_RE) 
    self.Sf  = np.zeros((nx,nf),dtype=DTYPE_COMP) 

    self.Pdf   = np.zeros((2,nx,nf),dtype=DTYPE_COMP)  # as this is actually (dp/dx, dp/dy) in fourier
    self.Visf  = np.zeros((2,nx,nf),dtype=DTYPE_COMP)
    self.UxWf  = np.zeros((2,nx,nf),dtype=DTYPE_COMP) #better than creating these at each and every time step
    self.UxW   = np.zeros((2,nx,ny),dtype=DTYPE_RE)

    self.K    = np.zeros((2,nx,nf))
    self.K_sq = np.zeros((nx,nf))
    self.dealias = np.zeros((nx,nf))
    self.K_K_sq =  np.zeros((2,nx,nf))

    self.fou_filt =  np.zeros((nx,nf))

    self.a_targ = [np.Inf]
    self.padx = 100
    self.pady = 100

  def transform_physical_fourier(self):
    pass

  def transform_fourier_physical(self):
    pass

def fourier_transform_2D(inp_var,out_var):
    """
    function to find the 2D fourier transform of the physical variable
    """
    out_var  = rfft2(inp_var)
    return out_var
    
def inv_fourier_transform_2D(inp_var,out_var):
    """
     function to find the 2D inverse fourier transform of the physical variable
    """
    out_var  = irfft2(inp_var)
    return out_var

def generate_mesh(SV):
  x = np.arange(SV.xR[0],SV.xR[1],SV.dx,dtype = dtype_re)
  y = np.arange(SV.yR[0],SV.yR[1],SV.dy,dtype = dtype_re)
  SV.Mesh[0],SV.Mesh[1] = np.meshgrid(x,y,indexing='ij')

def generate_fourier_space_vars(SV):
  kx = fftfreq(SV.nx,1/SV.nx)*2*PI/(SV.dx*SV.nx)
  kx[0] = 1.0e-16
  nf = int(SV.ny/2+1)
  ky = fftfreq(SV.ny,1/SV.ny)*2*PI/(SV.dy*SV.ny)
  ky[0] = 1.0e-16
  nf = int(SV.ny/2+1)
  ky = ky[:nf]; ky[-1]*=-1

  SV.K[0],SV.K[1] = np.array(np.meshgrid(kx,ky,indexing='ij'))

  SV.K_sq   = np.sum(np.square(SV.K),axis=0)

  kmax_2b3_x = SV.nx/3*2*PI/(SV.dx*SV.nx) 
  kmax_2b3_y = SV.ny/3*2*PI/(SV.dy*SV.ny)  

  SV.dealias = np.array(
      (np.abs(SV.K[0])<kmax_2b3_x)*
      (np.abs(SV.K[1])<kmax_2b3_y),
      dtype = bool
      )
  SV.K_K_sq = SV.K.astype(dtype_re)/np.where(SV.K_sq==0,1,SV.K_sq).astype(dtype_re)

  alpha = 36
  m = 36
  SV.fou_filt = np.exp(-alpha*(2*np.abs(SV.K[0])/SV.nx)**m)*np.exp(-alpha*(2*np.abs(SV.K[1])/SV.ny)**m)

def solve_RK4(SV,CBV):  
  a_targ_ind = 0
  # RK-4 vars
  aa = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
  bb = np.array([0,0.5,0.5,1])
 
  for i in range(2): SV.Uf[i] = rfft2(SV.U[i])
  for i in range(2): SV.Ut[i,:,:,0] = SV.U[i]

  # compute wavespace vectors
  generate_fourier_space_vars(SV)
  SV.it = 0
  Callbacks(SV,CBV)
  
  for it in range(1,SV.nt+1):
    SV.Uf_ = np.copy(SV.Uf)
    for rk in range(4):
      if rk>0:
        SV.Uf2_ = SV.Uf_ + SV.dUf*bb[rk]*SV.dt
      else:
        SV.Uf2_ = np.copy(SV.Uf_)

      vortexPairDriftVelCorrec(SV,CBV)

      computeRHS(SV)
      SV.Uf = SV.Uf + aa[rk]*SV.dt*SV.dUf

    SV.it = it
    Callbacks(SV,CBV)

    if it%SV.anim_s ==0:
      print("Iteration == ", it)
      for i in range(2): SV.Ut[i,:,:,int(it/SV.anim_s)] = irfft2(SV.Uf[i])
    
    # ComputeAllFlowQuantities()
    # computeKE(SV,CBV)

  for i in range(2): SV.Ut[i,:,:,-1] = irfft2(SV.Uf[i])