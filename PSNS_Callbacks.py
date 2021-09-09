import numpy as np
from DataTypes import *
from scipy.integrate import simps
import matplotlib.pyplot as plt


try:
  from pyfftw.interfaces.numpy_fft import fft2, ifft2, irfft2, rfft2
  import pyfftw
  pyfftw.interfaces.cache.enable()
  pyfftw.interfaces.cache.set_keepalive_time(60)
 #  pyfftw.config.NUM_THREADS = 4
  pyfftw.config.PLANNER_EFFORT = 'FFTW_MEASURE'
  # pyfftw.config.PLANNER_EFFORT = 'FFTW_ESTIMATE'
except ImportError:
  print("Couldn't load pyfftw libraries, resorting back to numpy fft libraries")
  pass # Rely on numpy . fft routines

class CallbackVars:
  def __init__(self,SV):
    self.enstropy                = np.zeros((SV.nt+1),dtype=dtype_re)               
    self.kinetic_energy          = np.zeros((SV.nt+1),dtype=dtype_re)
    self.max_val_residual        = np.zeros((2,SV.nt+1),dtype=dtype_re)
    self.max_val_residual_mframe = np.zeros(SV.nt+1,dtype=dtype_re)
    self.vortex_pair_cent        = np.zeros((2,2,SV.nt+1),dtype=dtype_re)
    self.vortex_pair_cent_ind    = np.zeros((2,2,SV.nt+1),dtype=dtype_re)
    self.vortex_pair_rad         = np.zeros((2,2,SV.nt+1),dtype=dtype_re)
    self.vortex_pair_drift_vel   = np.zeros((2,SV.nt+1),dtype=dtype_re)
    self.eulerResidual           = np.zeros((2,SV.nt+1),dtype=dtype_re)
    self.gamma                   = np.zeros((SV.nt+1),dtype=dtype_re)

    self.temp1                   = np.zeros((SV.nt+1),dtype=dtype_re)
    self.temp2                   = np.zeros((SV.nt+1),dtype=dtype_re)

def Callbacks(SV,CBV):
    # pass
    # computeKE(SV,CBV)
    # computeEnstropy(SV,CBV)
    computeResidual(SV,CBV)
    computeVortexPos(SV,CBV)
    # vortexPairDriftVel(SV,CBV)
    # vortexPairDriftVelCorrec(SV,CBV)  
    vortexPairRadius(SV,CBV)
    vortexPairEulerResidue(SV,CBV)
    # print('was here')
    
def areaAverage(Q,SV):

    dXx = SV.Mesh[0][1:SV.nx,1:SV.ny] - SV.Mesh[0][0:SV.nx-1,0:SV.ny-1]
    dXy = SV.Mesh[1][1:SV.nx,1:SV.ny] - SV.Mesh[1][0:SV.nx-1,0:SV.ny-1]
    A_i = dXx*dXy
    A   = np.sum(A_i)

    #value of Q at cell center
    Q_cell_cent = 0.25*(Q[0:SV.nx-1,0:SV.ny-1] + 
                        Q[0:SV.nx-1,1:SV.ny]   +  
                        Q[1:SV.nx,0:SV.ny-1]   + 
                        Q[1:SV.nx,1:SV.ny])
                                            
    avg = np.sum(Q_cell_cent*A_i)/A
    # avg = np

    return avg

def computeKE(SV,CBV):
  U = irfft2(SV.Uf)
  U_sq_2 = 0.5*(np.square(U[0]) + np.square(U[1]))
  CBV.kinetic_energy[SV.it] = areaAverage(U_sq_2,SV)

def computeEnstropy(SV,CBV):
  SV.W = PSNS_2D.vorticity2D(SV.Uf,SV)
  W_sq_2 = 0.5*np.square(SV.W)
  CBV.enstropy[SV.it] = areaAverage(W_sq_2,SV)

def computeResidual(SV,CBV):
    CBV.max_val_residual[0,SV.it] =  np.max(np.abs(irfft2(SV.Uf)-irfft2(SV.Uf_)))
    CBV.max_val_residual[1,SV.it] =  np.abs(np.max(irfft2(SV.Uf))-np.max(irfft2(SV.Uf_)))

def vortexPairCenter(Uf,SV):
    SV.W = PSNS_2D.vorticity2D(Uf,SV)
    # U = irfft2(Uf)
    # SV.W = PSNS_2D.vorticity2D_fd(U,SV)
    padx  = SV.padx
    pady  = SV.pady

    indx1 = int(SV.nx/2-padx)
    indx2 = int(SV.nx/2+padx)
    indy1 = 0
    indy2 = SV.ny

    W_temp1 = np.abs(SV.W[indx1:int(SV.nx/2),indy1:indy2])
    W_temp2 = np.abs(SV.W[int(SV.nx/2):indx2,indy1:indy2])

    x = np.zeros(2)
    y = np.zeros(2)
    x_ind = np.zeros(2)
    y_ind = np.zeros(2)

    x_ind[0] = np.unravel_index(W_temp1.argmax(), W_temp1.shape)[0] + indx1  
    y_ind[0] = np.unravel_index(W_temp1.argmax(), W_temp1.shape)[1] + indy1  
    x_ind[1] = np.unravel_index(W_temp2.argmax(), W_temp2.shape)[0] + int(SV.nx/2)  
    y_ind[1] = np.unravel_index(W_temp2.argmax(), W_temp2.shape)[1] + indy1  

    x[0] = SV.dx*(x_ind[0])
    y[0] = SV.dy*(y_ind[0])
    x[1] = SV.dx*(x_ind[1])
    y[1] = SV.dy*(y_ind[1])

    return np.array((x,y)),np.array((x_ind,y_ind))

def computeVortexPos(SV,CBV):
    # CBV.vortex_pair_cent[:,:,SV.it],CBV.vortex_pair_cent_ind[:,:,SV.it] = vortexPairCenter(SV.Uf,SV)
    CBV.vortex_pair_cent[:,:,SV.it],CBV.vortex_pair_cent_ind[:,:,SV.it] = vortexPairCenterVortCentroid(SV.Uf,SV)

def areaIntegral(Q,SV,limits):
  # maybe use scipy integrals
  nx1 = limits[0,0]
  nx2 = limits[0,1]
  ny1 = limits[1,0]
  ny2 = limits[1,1]

  dXx = SV.Mesh[0,nx1+1:nx2,ny1+1:ny2] - SV.Mesh[0,nx1:nx2-1,ny1:ny2-1]
  dXy = SV.Mesh[1,nx1+1:nx2,ny1+1:ny2] - SV.Mesh[1,nx1:nx2-1,ny1:ny2-1]
  A_i = dXx*dXy
  # print(nx1,nx2,ny1,ny2)

  #value of Q at cell center
  Q_cell_cent = 0.25*(Q[nx1:nx2-1,ny1:ny2-1]  + 
                      Q[nx1:nx2-1,ny1+1:ny2]  +  
                      Q[nx1+1:nx2,ny1:ny2-1]  + 
                      Q[nx1+1:nx2,ny1+1:ny2])
                                          
  integral = np.sum(Q_cell_cent*A_i)

  return integral

def vortexPairCenterVortCentroid(Uf,SV):
  nx_2 = int(SV.nx/2)
  x = np.zeros(2)
  y = np.zeros(2)
  x_ind = np.zeros(2)
  y_ind = np.zeros(2)

  _,Xcind = vortexPairCenter(Uf,SV)

  SV.W = PSNS_2D.vorticity2D(Uf,SV)
  # SV.U = irfft2(Uf)
  # SV.W = PSNS_2D.vorticity2D_fd(SV.U,SV)

  pady = SV.padx
  padx = SV.pady
  x1 = int(0.5*SV.nx-padx)
  x2 = int(0.5*SV.nx)
  y1 = int(Xcind[1,0]-pady*0.5)
  y2 = int(Xcind[1,0]+pady*0.5)

  
  vorticity_x_moment = SV.W[:,:]*SV.Mesh[0,:,:]
  vorticity_y_moment = SV.W[:,:]*SV.Mesh[1,:,:]

  # left plane integrals
  limits = np.array([[x1,x2],[y1,y2]])
  try:
    vorticity_x_moment_1_int = areaIntegral(vorticity_x_moment,SV,limits)
    vorticity_y_moment_1_int = areaIntegral(vorticity_y_moment,SV,limits)
    Gamma = areaIntegral(SV.W[:,:],SV,limits)
  except:
    print(limits)

  
  x[0] = vorticity_x_moment_1_int/Gamma
  y[0] = vorticity_y_moment_1_int/Gamma

  # right plane integrals
  x1 = int(0.5*SV.nx)
  x2 = int(0.5*SV.nx+padx)
  y1 = int(Xcind[1,1]-pady*0.5)
  y2 = int(Xcind[1,1]+pady*0.5)
  

  limits = np.array([[x1,x2],[y1,y2]])
  try:
    vorticity_x_moment_2_int = areaIntegral(vorticity_x_moment,SV,limits)
    vorticity_y_moment_2_int = areaIntegral(vorticity_y_moment,SV,limits)
    Gamma = areaIntegral(SV.W[:,:],SV,limits)
  except:
    print(limits)
  
  x[1] = vorticity_x_moment_2_int/Gamma
  y[1] = vorticity_y_moment_2_int/Gamma


  # find the indices close to the vortex centers
  x_ind[0] = np.abs(SV.Mesh[0,:,1] - x[0]).argmin()
  y_ind[0] = np.abs(SV.Mesh[1,1,:] - y[0]).argmin()
  x_ind[1] = np.abs(SV.Mesh[0,:,1] - x[1]).argmin()
  y_ind[1] = np.abs(SV.Mesh[1,1,:] - y[1]).argmin()

  return np.array((x,y)),np.array((x_ind,y_ind))

def vortexPairRadius(SV,CBV):
  # print('Was here')
  Uf = np.copy(SV.Uf)
  nx_2 = int(SV.nx/2)
  ax = np.zeros(2)
  ay = np.zeros(2)
  Xc,Xcind = vortexPairCenterVortCentroid(Uf,SV)
  # Xc,_ = vortexPairCenter(Uf,SV)

  SV.W = PSNS_2D.vorticity2D(Uf,SV)
  # SV.U = irfft2(Uf)
  # SV.W = PSNS_2D.vorticity2D_fd(SV.U,SV)

  pady = SV.padx
  padx = SV.pady
  x1 = int(0.5*SV.nx-padx)
  x2 = int(0.5*SV.nx)
  y1 = int(Xcind[1,0]-pady*0.5)
  y2 = int(Xcind[1,0]+pady*0.5)

  # left plane integrals
  vorticity_x_moment = SV.W[:,:]*np.square(SV.Mesh[0,:,:]- Xc[0,0])
  vorticity_y_moment = SV.W[:,:]*np.square(SV.Mesh[1,:,:]- Xc[1,0])
  limits = np.array([[x1,x2],[y1,y2]])
  vorticity_x_moment_1_int = areaIntegral(vorticity_x_moment,SV,limits)
  vorticity_y_moment_1_int = areaIntegral(vorticity_y_moment,SV,limits)
  Gamma = areaIntegral(SV.W[:,:],SV,limits)
  ax[0] = np.sqrt(np.abs(vorticity_x_moment_1_int/Gamma))
  ay[0] = np.sqrt(np.abs(vorticity_y_moment_1_int/Gamma))

  CBV.gamma[SV.it] = Gamma

  # right plane integrals
  x1 = int(0.5*SV.nx)
  x2 = int(0.5*SV.nx+padx)

  vorticity_x_moment = SV.W[:,:]*np.square(SV.Mesh[0,:,:]- Xc[0,1])
  vorticity_y_moment = SV.W[:,:]*np.square(SV.Mesh[1,:,:]- Xc[1,1])
  limits = np.array([[x1,x2],[y1,y2]])
  vorticity_x_moment_2_int = areaIntegral(vorticity_x_moment,SV,limits)
  vorticity_y_moment_2_int = areaIntegral(vorticity_y_moment,SV,limits)
  Gamma = areaIntegral(SV.W[:,:],SV,limits)
  ax[1] = np.sqrt(np.abs(vorticity_x_moment_2_int/Gamma))
  ay[1] = np.sqrt(np.abs(vorticity_y_moment_2_int/Gamma))

  CBV.vortex_pair_rad[0,0,SV.it] = ax[0]
  CBV.vortex_pair_rad[1,0,SV.it] = ay[0]
  CBV.vortex_pair_rad[0,1,SV.it] = ax[1]
  CBV.vortex_pair_rad[1,1,SV.it] = ay[1]

  return np.array((ax,ay))

def vortexPairDriftVel(SV,CBV):

  if SV.it>0:
    yn1   = CBV.vortex_pair_cent[1,0,SV.it]
    yn1_1 = CBV.vortex_pair_cent[1,0,SV.it-1]
    dvel1 = (yn1-yn1_1)/SV.dt

    yn2   = CBV.vortex_pair_cent[1,1,SV.it]
    yn2_1 = CBV.vortex_pair_cent[1,1,SV.it-1]
    dvel2 = (yn2-yn2_1)/SV.dt

    CBV.vortex_pair_drift_vel[0,SV.it] = dvel1
    CBV.vortex_pair_drift_vel[1,SV.it] = dvel2

def vortexPairDriftVelCorrec(SV,CBV):
  # xc_ind,yc_ind =int(CBV.vortex_pair_cent_ind[0,0,SV.it]),int(CBV.vortex_pair_cent_ind[1,0,SV.it])
  # Xc,Xcind = vortexPairCenter(SV.Uf2_,SV)   
  Xc,Xcind = vortexPairCenterVortCentroid(SV.Uf2_,SV)
  # xc_ind,yc_ind = int(Xcind[0,0]),int(Xcind[1,0])
  xc = Xc[0,0]
  yc = Xc[1,0]

  for i in range(2): SV.U[i,:,:] = irfft2(SV.Uf2_[i])
  vel = PSNS_2D.interpolate(xc,yc,SV.U[1],SV) 
  # print(vel)
  CBV.vortex_pair_drift_vel[0,SV.it] = vel
  CBV.vortex_pair_drift_vel[1,SV.it] = vel
  SV.U[1] = SV.U[1] - vel
  for i in range(2): SV.Uf2_[i] =  rfft2(SV.U[i])


  # vel = SV.U[1,xc_ind,yc_ind]
  # print('a==',vel)

def vortexPairEulerResidue(SV,CBV):
  # N = [<u.del omega>^2/<omega^2>]^1/2

  for i in range(2): SV.U[i,:,:] = irfft2(SV.Uf[i])

  Xc,Xcind = vortexPairCenterVortCentroid(SV.Uf,SV)

  SV.Wf = PSNS_2D.vorticity2D_f(SV.Uf,SV)
  SV.W  = PSNS_2D.vorticity2D(SV.Uf,SV)

  Wx_f  = 1j*SV.Wf*SV.K[0]*SV.fou_filt
  Wy_f  = 1j*SV.Wf*SV.K[1]*SV.fou_filt
  Wx    = irfft2(Wx_f)
  Wy    = irfft2(Wy_f)
  

  pady = SV.padx
  padx = SV.pady
  x1 = int(0.5*SV.nx-padx)
  x2 = int(0.5*SV.nx)
  y1 = int(Xcind[1,0]-pady*0.5)
  y2 = int(Xcind[1,0]+pady*0.5)

  # left vortex integrals
  Q = np.square(SV.U[0]*Wx + SV.U[1]*Wy)
  limits = np.array([[x1,x2],[y1,y2]])
  ER_numer = areaIntegral(Q,SV,limits)
  ER_denom = areaIntegral(np.square(SV.W),SV,limits)

  CBV.temp1[SV.it] = ER_numer
  CBV.temp2[SV.it] = ER_denom 

  CBV.eulerResidual[0,SV.it] = (ER_numer/ER_denom)**0.5

  # right vortex integrals
  x1 = int(0.5*SV.nx)
  x2 = int(0.5*SV.nx+padx)
  y1 = int(Xcind[1,1]-pady*0.5)
  y2 = int(Xcind[1,1]+pady*0.5)

  Q = np.square(SV.U[0]*Wx + SV.U[1]*Wy)
  limits = np.array([[x1,x2],[y1,y2]])
  ER_numer = areaIntegral(Q,SV,limits)
  ER_denom = areaIntegral(np.square(SV.W),SV,limits)

  CBV.eulerResidual[1,SV.it] = (ER_numer/ER_denom)**0.5
  # print((ER_numer/ER_denom)**0.5)

  A = np.sum(np.square(SV.U[0,x1:x2,y1:y2]*Wx[x1:x2,y1:y2] + SV.U[1,x1:x2,y1:y2]*Wy[x1:x2,y1:y2]))*SV.dx*SV.dy
  B = np.sum(np.square(SV.W[x1:x2,y1:y2]))*SV.dx*SV.dy

  file_out = open("Euler_Res.txt","a")
  file_out.write(str((ER_numer/ER_denom)**0.5)+" "+str((A/B)**0.5)+" "+str(ER_numer)+"\n")
  file_out.close()
  # print(limits)


def vortexPairGamma(SV,CBV):
  pass

import PSNS_2D 
