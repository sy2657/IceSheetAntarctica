
         
# Physics 2D equation
def IceSheet2D(u,v,s,H,B, x, y):

  g= 9.8
  rho = 917

  n = 3 # what is n? 

  Y = tf.concat([u, v, s, H, B], 1)

  Y_x = fwd_gradients(Y, x)
  Y_y = fwd_gradients(Y, y)
  Y_xx = fwd_gradients(Y_x, x)
  Y_yy = fwd_gradients(Y_y, y)

  Y_xy = fwd_gradients(Y_x, y)
  Y_yx = fwd_gradients(Y_y, x)

  u = Y[:,0:1]
  v = Y[:,1:2]
  s = Y[:,2:3]
  H = Y[:,3:4]
  B = Y[:,4:5]

  u_x = Y_x[:,0:1]
  v_x = Y_x[:,1:2]
  s_x = Y_x[:,2:3]
  H_x = Y_x[:,3:4]
  B_x = Y_x[:,4:5]

  u_y = Y_y[:,0:1]
  v_y = Y_y[:,1:2]
  s_y = Y_y[:,2:3]
  H_y = Y_y[:,3:4]
  B_y = Y_y[:,4:5]

  u_xx = Y_xx[:,0:1]
  v_xx = Y_xx[:,1:2]
  s_xx = Y_xx[:,2:3]
  H_xx = Y_xx[:,3:4]
  B_xx = Y_xx[:,4:5]

  # mixed derivatives
  u_yx = Y_yx[:, 0:1]
  v_yx = Y_yx[:, 1:2]

  u_xy = Y_xy[:, 0:1]
  v_xy = Y_xy[:, 1:2]

  u_yy = Y_yy[:,0:1]
  v_yy = Y_yy[:,1:2]
  s_yy = Y_yy[:,2:3]
  H_yy = Y_yy[:,3:4]
  B_yy = Y_yy[:,4:5]

  # define mu
  

  #mu = 0.5*B*(pow( 0.5(pow( u_x, 2) + pow(v_y, 2)) + pow(0.5( u_y + v_x), 2) , 0.5(1 - (1/n))))
  mu = 0.5*B*(pow(0.5*(pow( u_x, 2) + pow(v_y, 2)) + pow(0.5*( u_y + v_x), 2) , 0.5*(1 - (1/n))))

  mu1 = 0.5*(pow( u_x, 2) + pow(v_y, 2)) + pow(0.5*( u_y + v_x), 2) 
  mu_x = 0.5*B*(0.5*(1 - (1/n)))*( pow(mu1, 0.5*(1 - (1/n)) - 1))*(0.5*( 2*u_x*u_xx + 2*v_y*v_yx) + 0.5*( u_y +v_x)*(u_yx + v_xx))

  mu_y = 0.5*(B)*(0.5*(1 - (1/n)))*( pow(mu1, 0.5*(1 - (1/n)) - 1))*(0.5*(2*u_x*u_xy + 2*v_y*v_yy) + 0.5*(u_y + v_x)*(u_yy + v_xy))
  # temp constants
  #mu = B/2
  #mu = 1
  #mu_x = 1
  #mu_y = 1

  e1term1 = (H_x + H)*(4*u_x + 2*v_y) + H*mu*(4*u_xx + 2*v_yx)
  e1term2 = (H_y*mu + mu_y*H)*(u_y + v_x) + H*mu*(u_yy + v_xy)
  e1term3 = rho*g*H*s_x

  e2term1 = (H_y*mu + mu_y*H)*(4*v_y + 2*u_x) + H*mu*(4*v_yy + 2*u_xy )
  e2term2 = (H_x*mu + mu_x*H)*(u_y + v_x) + H*mu*(u_yx +v_xx)
  e2term3 = rho*g*H*s_y

  e1 = e1term1 + e1term2 + e1term3
  e2 = e2term1 + e2term2 +e2term3
  e3 = fwd_gradients(H*u, x) + fwd_gradients(H*v, y)

  return e1, e2, e3
