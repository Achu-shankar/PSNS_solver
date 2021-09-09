from matplotlib.pyplot import colorbar
from PSNS_2D import *
from PSNS_Callbacks import CallbackVars, Callbacks,computeKE,vortexPairCenter
from numpy import pi, cos, sin,exp
import pickle
from numpy import savetxt
import pandas as pd

# def initial_condition(SV):
#     sL = 2*pi
#     d_w = 80
#     d_p = 0.05
#     mask = SV.Mesh[1]<= 0.5
#     n_mask = np.invert(mask)
#     mask = mask.astype(float) 
#     n_mask = n_mask.astype(float) 
#     SV.U[0] = mask*np.tanh(d_w*(SV.Mesh[1]-0.25)) + n_mask*np.tanh(d_w*(0.75-SV.Mesh[1]))
#     SV.U[1] = d_p*sin(2*pi*(SV.Mesh[0]))
    

def initial_condition(SV):
    X = SV.Mesh
    sL = SV.Lx
    xc1 =  0.5
    xc2 = -0.5
    a_sq = 0.4**2
    x1    = (X[0]-0.5*sL - xc1)
    x2    = (X[0]-0.5*sL - xc2)
    y1    =  X[1] - 0.5*sL
    y2    =  X[1] - 0.5*sL
    r_sq1 = np.square(x1) + np.square(y1) + 1e-6
    r_sq2 = np.square(x2) + np.square(y2) + 1e-6

    SV.U[0]  = -(y1*(1-exp(-r_sq1/a_sq))/r_sq1 - y2*(1-exp(-r_sq2/a_sq))/r_sq2)
    SV.U[1]  =  (x1*(1-exp(-r_sq1/a_sq))/r_sq1 - x2*(1-exp(-r_sq2/a_sq))/r_sq2) 
    # SV.U[0]  = -(y1*(1-exp(-r_sq1/a_sq))/r_sq1)
    # SV.U[1]  =  (x1*(1-exp(-r_sq1/a_sq))/r_sq1) 
    # SV.U[1]  =  (x1*(1-exp(-r_sq1/a_sq))/r_sq1 - x2*(1-exp(-r_sq2/a_sq))/r_sq2) + (1-exp(-1/a_sq))

def LC_initial_condition(SV):
    X = SV.Mesh
    sL = SV.Lx
    xc1 =  0.5
    xc2 = -0.5
    a_sq = 0.01
    r_sq1 = np.square(X[0]-0.5*sL - xc1) + np.square(X[1]-0.5*sL*3) + 1e-6
    r_sq2 = np.square(X[0]-0.5*sL - xc2) + np.square(X[1]-0.5*sL*3) + 1e-6
    x1    = (X[0]-0.5*sL - xc1)
    x2    = (X[0]-0.5*sL - xc2)
    y1    =  X[1] - 0.5*sL*3
    y2    =  X[1] - 0.5*sL*3

    SV.U[0]  = -(y1*(1-exp(-r_sq1/a_sq))/r_sq1 - y2*(1-exp(-r_sq2/a_sq))/r_sq2)
    SV.U[1]  =  (x1*(1-exp(-r_sq1/a_sq))/r_sq1 - x2*(1-exp(-r_sq2/a_sq))/r_sq2) 
    # SV.U[1]  =  (x1*(1-exp(-r_sq1/a_sq))/r_sq1 - x2*(1-exp(-r_sq2/a_sq))/r_sq2) + (1-exp(-1/a_sq))

def vortex_pair_data_extractor(SV):

    # only going to save the data for right vortex 
    # Nt = -1
    # Ut = SV.Ut[:,:,:,Nt]
    # Uf = rfft2(Ut)
    Xc,Xc_ind = vortexPairCenter(SV.Uf,SV)  # SV.Uf has the value for last time step
    xc = Xc[0,1]
    yc = Xc[1,1]
    xc_ind = Xc_ind[0,1]
    yc_ind = Xc_ind[1,1]

    xc_middle_plane = 0.5*(Xc[0,0]+Xc[0,1])
    padx = 2
    pady = 2

    # size of box center around the middle plane of the pair
    box_w = 2  # in length units 
    box_h = 4   # in length units
    box_w_ind = int(box_w/SV.dx)
    box_h_ind = int(box_h/SV.dy)

    x_ind1 = int(xc_ind - int(0.5*np.abs(Xc[0,0]-Xc[0,1])/SV.dx) - padx)
    x_ind2 = int(x_ind1 + box_w_ind   + padx)
    y_ind1 = int(yc_ind - box_h_ind/2 - pady)
    y_ind2 = int(yc_ind + box_h_ind/2 + pady)

    U = irfft2(SV.Uf)
    # print(xc_ind,x_ind1,x_ind2,y_ind1,y_ind2,yc_ind)
    # print(Xc,Xc_ind)
    # print(Xc[0,0]-Xc[0,1])

    U_extract    = np.copy(U[:,x_ind1:x_ind2,y_ind1:y_ind2])
    Mesh_extract = np.copy(SV.Mesh[:,x_ind1:x_ind2,y_ind1:y_ind2])
    # # print(Mesh_extract.shape)
    Mesh_extract[0] = Mesh_extract[0] - xc_middle_plane
    Mesh_extract[1] = Mesh_extract[1] - yc

    dudx, dudy, dvdx, dvdy = velGradient(SV.Uf,SV)
    dudx_e, dudy_e, dvdx_e, dvdy_e = dudx[x_ind1:x_ind2,y_ind1:y_ind2].flatten(), \
                                     dudy[x_ind1:x_ind2,y_ind1:y_ind2].flatten(), \
                                     dvdx[x_ind1:x_ind2,y_ind1:y_ind2].flatten(), \
                                     dvdy[x_ind1:x_ind2,y_ind1:y_ind2].flatten()

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    s_x = np.linspace(0,1,10)
    s_y = np.zeros(s_x.shape)
    start_points = np.array((s_x,s_y)).T
    print(start_points.shape)
    plt.streamplot(Mesh_extract[0].T,Mesh_extract[1].T, U_extract[0].T, U_extract[1].T,\
     density=[20,20],color='k',start_points=start_points,minlength=0.1)
    plt.contourf(Mesh_extract[0],Mesh_extract[1],U_extract[0],levels = 30)
    plt.colorbar()
    plt.show()

    x = Mesh_extract[0].flatten()
    y = Mesh_extract[1].flatten()
    u = U_extract[0].flatten()
    v = U_extract[1].flatten()

    df = pd.DataFrame({"x" : x, "y" : y,"u" : u, "v" : v,"dudx":dudx_e,"dudy":dudy_e,"dvdx":dvdx_e,"dvdy":dvdy_e})
    df.to_csv("vel.csv", index=False)

def base_flow_constraint(SV):
    U = irfft2(SV.Uf*SV.dealias)
    SV.Wf = vorticity2D_f(SV.Uf,SV)
    Wx    = irfft2(1j*SV.K[0]*SV.Wf*SV.dealias) 
    Wy    = irfft2(1j*SV.K[1]*SV.Wf*SV.dealias) 

    residual = U[0]*Wx + U[1]*Wy
    # residual = Wy
    return residual

t1 = time.time()
SV = SolverVars(xR =[0,2*pi],yR = [0,2*pi],nx=256,ny=256,dt=0.001,T=1,anim_s=100,nu=0.00001)
CBV = CallbackVars(SV)
generate_mesh(SV)
wavespace(SV)
initial_condition(SV)
solve_RK4(SV,CBV)
print("Total simulation time == ",time.time()-t1)
# vortex_pair_data_extractor(SV)
res = base_flow_constraint(SV)
print(np.sum(np.abs(res)),np.max(res))
plt.contourf(SV.Mesh[0],SV.Mesh[1],res)
plt.clim(0,100)
plt.colorbar()
plt.show()
# plt.plot(CBV.vortex_pair_cent[0,0,:])
# plt.plot(CBV.vortex_pair_cent[0,1,:])
# plt.show()
# plt.plot(CBV.vortex_pair_cent[1,0,:])
# plt.plot(CBV.vortex_pair_cent[1,1,:])
# plt.show()

# plt.plot(CBV.vortex_pair_rad[0,0,:])
# # plt.plot(CBV.vortex_pair_rad[0,1,:])
# plt.show()
# plt.plot(CBV.vortex_pair_rad[1,0,:])
# # plt.plot(CBV.vortex_pair_rad[1,1,:])
# plt.show()


# save(SV,CBV)


# # plt.plot(CBV.kinetic_energy)
# # plt.show()
# # plt.plot(CBV.enstropy)
# # plt.show()
# # plt.plot(CBV.max_val_residual)
# # plt.show()
# plt.plot(CBV.vortex_pair_cent[0,0,:])
# plt.plot(CBV.vortex_pair_cent[0,1,:])
# # plt.show()
# plt.plot(CBV.vortex_pair_cent[1,0,:])
# plt.plot(CBV.vortex_pair_cent[1,1,:])
# plt.show()

Nt = -1
Ut = SV.Ut[:,:,:,Nt]
# Ut = SV.U
Uf = rfft2(Ut)
W = np.zeros((SV.nx,SV.ny),dtype=dtype_re)
W = vorticity2D(Uf,SV)
S = streamfunction(Uf,SV)
W_d = vorticity2D_fd(Ut,SV)
# W[W<0] = 0
# W_d[W_d<0] = 0

# print(np.sum(np.abs(W)),np.min(W),np.max(W))
# print(np.sum(np.abs(W_d)),np.min(W),np.max(W))

fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.contourf(SV.Mesh[0],SV.Mesh[1],W,levels = 20)
plt.plot(CBV.vortex_pair_cent[0,0,-1],CBV.vortex_pair_cent[1,0,-1],'ro',markersize=2)
plt.plot(CBV.vortex_pair_cent[0,1,-1],CBV.vortex_pair_cent[1,1,-1],'ro',markersize=2)

plt.colorbar()
plt.show()

# print(CBV.vortex_pair_rad)

# print(np.sum(np.abs(res[SV.Mesh[1]>4.5])))
# x =SV.Mesh[1,:,1]
# y = 
# fig, ax = plt.subplots()
# ax.set_aspect('equal')
# plt.contourf(SV.Mesh[0],SV.Mesh[1],Ut[1],levels = 20)
# # # plt.streamplot(SV.Mesh[0,:,:].T,SV.Mesh[1,:,:].T, SV.Ut[0,:,:,Nt].T, SV.Ut[1,:,:,Nt].T, density=[2, 2],color='k')
# plt.show()



# # Nt = -1
# # Ut = SV.Ut[:,:,:,Nt]
# # Uf = rfft2(Ut)
# # W = np.zeros((SV.nx,SV.ny),dtype=dtype_re)
# # W = vorticity2D(Uf,SV)
# # S = streamfunction(Uf,K,K_sq)


# # fig, ax = plt.subplots()

# # ax.set_aspect('equal')

# # plt.contourf(X[0],X[1],S,levels = 30)
# # plt.colorbar()


# filename = r'E:/OneDrive/Research/code/PsuedoSpecNS/ouputs/data.pkl'
# open_file = open(filename, "rb")
# A = pickle.load(open_file)
# B = pickle.load(open_file)
# open_file.close()
# print(A.nx,B.enstropy)