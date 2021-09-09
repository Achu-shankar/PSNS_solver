import numpy as np
from numpy import pi, cos, sin
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy.fft import fftfreq, fft2, ifft2, irfft2, rfft2 
import time
import pickle 
# from numba import jit, njit
# from accelerate import profiler
from PSNS_Callbacks import CallbackVars, Callbacks,computeKE
from DataTypes import *



try:
  from pyfftw.interfaces.numpy_fft import fft2, ifft2, irfft2, rfft2
  import pyfftw
  pyfftw.interfaces.cache.enable()
  pyfftw.interfaces.cache.set_keepalive_time(60)
  pyfftw.config.NUM_THREADS = 4
  pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
  # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
except ImportError:
  print("Couldn't load pyfftw libraries, resorting back to numpy fft libraries")
  pass # Rely on numpy . fft routines


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
    self.it = 1

    # nf  = ny
    nf = int(ny/2+1)
    self.Ut  = np.zeros((2,nx,ny,int(self.nt/anim_s+1)),dtype=dtype_re)
    self.U   = np.zeros((2,nx,ny),dtype=dtype_re)
    self.Uf  = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.Uf_ = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.Uf2_ = np.zeros((2,nx,nf),dtype=dtype_comp) # second intermediate value
    self.Uf_temp = np.zeros((2,nx,nf),dtype=dtype_comp)
    self.dUf = np.zeros((2,nx,nf),dtype=dtype_comp) 
    self.P   = np.zeros((nx,ny),dtype=dtype_re)
    self.Pf  = np.zeros((nx,nf),dtype=dtype_comp)
    self.W   = np.zeros((nx,ny),dtype=dtype_re) 
    self.Wf  = np.zeros((nx,nf),dtype=dtype_comp) 
    self.S   = np.zeros((nx,nf),dtype=dtype_re) 
    self.Sf  = np.zeros((nx,nf),dtype=dtype_comp) 

    self.Pdf   = np.zeros((2,nx,nf),dtype=dtype_comp)  # as this is actually (dp/dx, dp/dy) in fourier
    self.Visf  = np.zeros((2,nx,nf),dtype=dtype_comp)
    self.UxWf  = np.zeros((2,nx,nf),dtype=dtype_comp) #better than creating these at each and every time step
    self.UxW   = np.zeros((2,nx,ny),dtype=dtype_re)

    self.K    = np.zeros((2,nx,nf))
    self.K_sq = np.zeros((nx,nf))
    self.dealias = np.zeros((nx,nf))
    self.K_K_sq =  np.zeros((2,nx,nf))

    # self.Wf  = pyfftw.empty_aligned((nx,nf), dtype='complex64')
    # self.W   = pyfftw.empty_aligned((nx,ny),dtype='float32') 
    # self.UxWf  = pyfftw.empty_aligned((2,nx,nf),dtype='complex64') #better than creating these at each and every time step
    # self.UxW   = pyfftw.empty_aligned((2,nx,ny),dtype='float32')

    # self.W_ifft_obj = pyfftw.FFTW(self.Wf,self.W, axes=(0,1),direction='FFTW_BACKWARD')
    # self.UxW_fft_obj = pyfftw.FFTW(self.UxW, self.UxWf, axes=(1,2),direction='FFTW_FORWARD')
    # self.U_nonLin_ifft_obj = pyfftw.FFTW(self.Uf_temp,self.U, axes=(1,2),direction='FFTW_BACKWARD')

    # self.W_ifft_obj  = pyfftw.builders.irfft2(self.Wf)
    # self.UxW_fft_obj = pyfftw.builders.rfft2(self.UxW)
    # self.U_nonLin_ifft_obj = pyfftw.builders.irfft2(self.Uf_temp)

def solve_RK4(SV,CBV):  
  # RK-4 vars
  aa = np.array([1.0/6.0,1.0/3.0,1.0/3.0,1.0/6.0])
  bb = np.array([0,0.5,0.5,1])
 
  for i in range(2): SV.Uf[i] = rfft2(SV.U[i])
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
      for i in range(2): SV.Ut[i,:,:,int(it/SV.anim_s)] = irfft2(SV.Uf[i])
    SV.it = it
    # ComputeAllFlowQuantities()

    Callbacks(SV,CBV)
    # computeKE(SV,CBV)

  for i in range(2): SV.Ut[i,:,:,int(it/SV.anim_s)] = irfft2(SV.Uf[i])
      
  # return SV,CB_vars

def wavespace(SV):
  kx = fftfreq(SV.nx,1/SV.nx)*2*pi/(SV.dx*SV.nx)
  kx[0] = 1.0e-16
  # ky = fftfreq(SV.ny,1/SV.ny)*2*pi/(SV.dy*SV.ny)
  # ky[0] = 1.0e-6
  nf = int(SV.ny/2+1)
  ky = fftfreq(SV.ny,1/SV.ny)*2*pi/(SV.dy*SV.ny)
  ky[0] = 1.0e-16
  nf = int(SV.ny/2+1)
  ky = ky[:nf]; ky[-1]*=-1

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
  SV.Uf_temp = SV.Uf2_*SV.dealias 
  # SV.Uf_temp = SV.Uf2_
  for i in range(2): SV.U[i] = irfft2(SV.Uf_temp[i])  
  # SV.U = SV.U_nonLin_ifft_obj(SV.Uf_temp)
  # SV.U[:,:,:] = SV.U_nonLin_ifft_obj()
  
  SV.W = vorticity2D(SV.Uf_temp,SV)
  # SV.U = irfft2(SV.Uf2_)
  # SV.W = vorticity2D_fd(SV.U,SV)
  
  SV.UxW[0]  =  SV.U[1]*SV.W
  SV.UxW[1]  = -SV.U[0]*SV.W
  SV.UxWf[0] =  rfft2(SV.UxW[0])
  SV.UxWf[1] =  rfft2(SV.UxW[1])
  # SV.UxWf    =  SV.UxW_fft_obj(SV.UxW)
  # SV.UxWf[:,:,:]    =  SV.UxW_fft_obj()
  SV.UxWf    =  SV.UxWf*SV.dealias 
  

def vorticity2D(Uf,SV): 
  SV.Wf = (1j*(SV.K[0]*Uf[1] - SV.K[1]*Uf[0]))
  SV.W  = irfft2(SV.Wf)
  # SV.W = SV.W_ifft_obj(SV.Wf)
  # SV.W[:,:] = SV.W_ifft_obj()
  return SV.W

def vorticity2D_fd(U,SV): 
  SV.W  = np.gradient(U[1],SV.dx,axis=0) - np.gradient(U[0],SV.dy,axis=1)
  return SV.W

def vorticity2D_f(Uf,SV): 
  SV.Wf = (1j*(SV.K[0]*Uf[1] - SV.K[1]*Uf[0]))
  return SV.Wf
# Computes Streamfunction
def streamfunction(Uf,SV): 
  Wf = np.zeros((SV.nx,SV.ny),dtype = dtype_comp)
  Wf = vorticity2D_f(Uf,SV)
  Sf = Wf/SV.K_sq
  S = irfft2(Sf)
  return S

def velGradient(Uf,SV):
  dudx = irfft2(1j*SV.K[0]*Uf[0])
  dudy = irfft2(1j*SV.K[1]*Uf[0])
  dvdx = irfft2(1j*SV.K[0]*Uf[1])
  dvdy = irfft2(1j*SV.K[1]*Uf[1])

  return dudx, dudy, dvdx, dvdy

def save(SV,CBV):
  filename = r'E:/OneDrive/Research/code/PsuedoSpecNS/ouputs/data.pkl'
  open_file = open(filename, "wb")
  pickle.dump(SV, open_file)
  pickle.dump(CBV, open_file)
  open_file.close()
