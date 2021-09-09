from PSNS_2D import solve_RK4, mesh
import numpy as np
from numpy import pi, cos, sin

# Apply initial condition
def initial_condition(U,X):
  U[0] = 0.2 +sin(X[0])*cos(X[1])
  U[1] = 0.2 -cos(X[0])*sin(X[1])
  return U

# user inputs
xR = [0,2*pi]
yR = [0,2*pi]
nx = 128
ny = 128
Lx = xR[1]-xR[0]
Ly = yR[1]-yR[0]
dx = Lx/nx
dy = Ly/ny
nu = 0.000625

T = 1
dt = 0.01
nt = int(T/dt)

X = mesh(dx,dy,xR,yR)

# K,K_sq,dealias = wavespace(nx,ny)

# t1 = time.time()

sol = solve_RK4(X,nx,ny,nt,dx,dy,dt,nu,initial_condition,si=1)