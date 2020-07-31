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

xmin = 1280
xmax = 1320
ymin = 1790
ymax = 1830


b = bmat[xmin:xmax, ymin:ymax]/1e6
h = hmat[xmin:xmax, ymin:ymax]/1e6
s = smat[xmin:xmax, ymin:ymax]/1e6
u = umat[xmin:xmax, ymin:ymax]/1e6
v = vmat[xmin:xmax, ymin:ymax]/1e6
x = xmat[xmin:xmax, ymin:ymax]/1e6
y = ymat[xmin:xmax, ymin:ymax]/1e6

# do not flatten
x_s = x.flatten()
#y_star = y.flatten() #flatten

N = x_s.shape[0]

# flatten the other variables
#u_star = u.flatten()
#v_star = v.flatten()
#s_star = s.flatten()
#b_star = b.flatten()
#h_star = h.flatten()

# reshape
x_star = x.reshape(N, 1)
y_star = y.reshape(N,1)
u_star = u.reshape(N,1)
v_star = v.reshape(N,1)
s_star = s.reshape(N,1)
b_star = b.reshape(N,1)
h_star = h.reshape(N,1)


T = 1

# rearrange
X_star = np.tile(x_star, (1,1))
Y_star = np.tile(y_star, (1,1))



## training data
N_data = N

idx_t = 1

idx_x = np.random.choice(N, N_data, replace=False) # rearrange the indices 

#xs = X_star[idx_x, :]
#x_data = xs.flatten()[:,None]


x_data = X_star[idx_x,:].flatten()[:,None]
y_data = Y_star[idx_x,:].flatten()[:,None]

u_data = u_star[idx_x, :].flatten()[:, None]
v_data = v_star[idx_x, :].flatten()[:, None]
s_data = s_star[idx_x, :].flatten()[:, None]
h_data = h_star[idx_x, :].flatten()[:, None]
b_data = b_star[idx_x, :].flatten()[:, None]

N_eqns = N

idx_x = np.random.choice(N, N_eqns, replace=False)

x_eqns = X_star[idx_x, :].flatten()[:,None]
y_eqns = Y_star[idx_x, :].flatten()[:,None]

#u_eqns = U_star[:, idx_x].flatten()[:, None]
#v_eqns = V_star[:, idx_x].flatten()[:, None]
#s_eqns = S_star[:, idx_x].flatten()[:, None]
#h_eqns = H_star[:, idx_x].flatten()[:, None]
#b_eqns = B_star[:, idx_x].flatten()[:, None]

