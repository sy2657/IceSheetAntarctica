# imports

import tensorflow as tf
import numpy as np
import scipy.io
import time
import sys

from utilities import neural_net, Navier_Stokes_2D, Gradient_Velocity_2D, tf_session, mean_squared_error, relative_error, fwd_gradients

# custom step 1

# num. hidden layer , num. hidden units 
# increase layers / units to decrease loss 

layers = [2] + 10*[5*20] +[5] # 2 input, 5 output

batch_size = 10000

data = scipy.io.loadmat('data_Serena.mat')

# normalize
bmat = data['B']
hmat = data['H']
smat = data['S']
umat = data['u']
vmat = data['v']
xmat = data['x']
ymat = data['y']

# take square subset of non-nan values

#xmin = 1280
#xmax = 1320
#ymin = 1790
#ymax = 1830
xmin= 1000
xmax = 2000
ymin= 1000
ymax = 2000

b = bmat[xmin:xmax, ymin:ymax]/1e6
h = hmat[xmin:xmax, ymin:ymax]/1e6
s = smat[xmin:xmax, ymin:ymax]/1e6
u = umat[xmin:xmax, ymin:ymax]/1e6
v = vmat[xmin:xmax, ymin:ymax]/1e6
x = xmat[xmin:xmax, ymin:ymax]/1e6
y = ymat[xmin:xmax, ymin:ymax]/1e6


x_star = x.flatten()
y_star = y.flatten() #flatten

# flatten the other variables
u_star = u.flatten()
v_star = v.flatten()
s_star = s.flatten()
b_star = b.flatten()
h_star = h.flatten()


# square dimensions 

N = x_star.shape[0]

#print(x_star.shape[0])
#N= x_star.shape[0]
T = 1

X_star = np.tile(x_star, (1,T)) # X_star.shape (1, 1600) : adds an extra bracket 
Y_star = np.tile(y_star, (1,T))

#U_star = np.tile(u_star, (1,T))
U_star = u.flatten()
V_star = v.flatten()
S_star = s.flatten()
H_star = h.flatten()
B_star = b.flatten()


## training data
N_data = N
#N_data = 1000
#N = 1000

#idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
idx_t = 1

idx_x = np.random.choice(N, N_data, replace=False) # rearrange the indices 
idx_y = np.random.choice(N, N_data, replace=False)

#x_data = x_star[:, idx_t][:, idx_x].flatten()[:,None]
# x_data = X_star[:, idx_x]
x_data = X_star[:, idx_x].flatten()[:,None]
y_data = Y_star[:, idx_x].flatten()[:,None]

u_data = U_star[:, idx_x].flatten()[:, None]
v_data = V_star[:, idx_x].flatten()[:, None]
s_data = S_star[:, idx_x].flatten()[:, None]
h_data = H_star[:, idx_x].flatten()[:, None]
b_data = B_star[:, idx_x].flatten()[:, None]

N_eqns = N

idx_x = np.random.choice(N, N_eqns, replace=False)

x_eqns = X_star[:, idx_x].flatten()[:,None]
y_eqns = Y_star[:, idx_x].flatten()[:,None]

u_eqns = U_star[:, idx_x].flatten()[:, None]
v_eqns = V_star[:, idx_x].flatten()[:, None]
s_eqns = S_star[:, idx_x].flatten()[:, None]
h_eqns = H_star[:, idx_x].flatten()[:, None]
b_eqns = B_star[:, idx_x].flatten()[:, None]
