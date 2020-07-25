
class HFMI(object): # hidden fluid mechanics ice

  def __init__(self, x_data, y_data, u_data, v_data, s_data, h_data,b_data,x_eqns, y_eqns, layers, batch_size):
    
    self.layers = layers
    self.batch_size = batch_size

    # data
    [self.x_data, self.y_data, self.u_data, self.v_data, self.s_data, self.h_data, self.b_data] = [x_data, y_data, u_data, v_data, s_data, h_data, b_data]
    [self.x_eqns, self.y_eqns] = [x_eqns, y_eqns]


    # change above to x, y, u,v,s,h ,b 
    [self.x_data_tf, self.y_data_tf, self.u_data_tf, self.v_data_tf, self.s_data_tf, self.h_data_tf, self.b_data_tf]  = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(7)]
    [self.x_eqns_tf, self.y_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(2)]

    # physics uninformed NNs 
    self.net_uvshb = neural_net(self.x_data, self.y_data, layers = self.layers)

    [self.u_data_pred,self.v_data_pred, self.s_data_pred, self.h_data_pred, self.b_data_pred] = self.net_uvshb(self.x_data_tf, self.y_data_tf)

    # physics informed NNs
    [self.u_eqns_pred, self.v_eqns_pred, self.s_eqns_pred, self.h_eqns_pred, self.b_eqns_pred] = self.net_uvshb(self.x_eqns_tf, self.y_eqns_tf)

    [self.e1_eqns_pred,self.e2_eqns_pred, self.e3_eqns_pred] = IceSheet2D(self.u_eqns_pred,
                                                  self.v_eqns_pred,
                                                  self.s_eqns_pred,
                                                  self.h_eqns_pred,
                                                  self.b_eqns_pred,
                                                  self.x_eqns_tf,
                                                  self.y_eqns_tf)

    self.loss = mean_squared_error(self.u_data_pred, self.u_data_tf) + mean_squared_error(self.v_data_pred, self.v_data_tf)+ mean_squared_error(self.s_data_pred, self.s_data_tf) + mean_squared_error(self.h_data_pred, self.h_data_tf)+ mean_squared_error(self.b_data_pred, self.b_data_tf) + mean_squared_error(self.e1_eqns_pred, 0.0) + mean_squared_error(self.e2_eqns_pred, 0.0) + mean_squared_error(self.e3_eqns_pred, 0.0) 
    self.learning_rate = tf.placeholder(tf.float32, shape=[])
    self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
    self.train_op = self.optimizer.minimize(self.loss)

    self.sess = tf_session()
  # train function
  def train(self, total_time, learning_rate):
    N_data = self.x_data.shape[0]
    N_eqns = self.x_eqns.shape[0]
    start_time = time.time()
    running_time = 0
    it = 0
    while running_time < total_time:
      idx_data = np.random.choice(N_data, min(self.batch_size, N_data))
      idx_eqns = np.random.choice(N_eqns, self.batch_size)

      (x_data_batch, y_data_batch, u_data_batch, v_data_batch, s_data_batch, h_data_batch, b_data_batch) = ( self.x_data[idx_data,:], self.y_data[idx_data,:], self.u_data[idx_data,:], self.v_data[idx_data,:], self.s_data[idx_data,:], self.h_data[idx_data,:], self.b_data[idx_data,:])

      # remove u v s h b 
      (x_eqns_batch,y_eqns_batch) = (self.x_eqns[idx_eqns,:],
                                self.y_eqns[idx_eqns,:])
      
      tf_dict = { self.x_data_tf: x_data_batch,
                  self.y_data_tf: y_data_batch,
                  self.u_data_tf: u_data_batch,
                  self.v_data_tf: v_data_batch,
                  self.s_data_tf: s_data_batch,
                  self.h_data_tf: h_data_batch,
                  self.b_data_tf: b_data_batch,
                  self.x_eqns_tf: x_eqns_batch,
                  self.y_eqns_tf: y_eqns_batch,
                  self.learning_rate: learning_rate}
      
      self.sess.run([self.train_op], tf_dict)

      if it > 50:
        break

      if it % 2 == 0: # change from: it % 10 to 
        elapsed = time.time() - start_time
        running_time += elapsed/3600.0
        [loss_value,
          learning_rate_value] = self.sess.run([self.loss,
                                                self.learning_rate], tf_dict)
        print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
              %(it, loss_value, elapsed, running_time, learning_rate_value))
        
        lossvalues.append(loss_value)
        iterations.append(it)
        sys.stdout.flush()
        start_time = time.time()
      it += 1
  def predict(self, x_star, y_star):
    tf_dict = {self.x_data_tf: x_star, self.y_data_tf: y_star}
    u_star = self.sess.run(self.u_data_pred, tf_dict)
    v_star = self.sess.run(self.v_data_pred, tf_dict)
    s_star = self.sess.run(self.s_data_pred, tf_dict)
    h_star = self.sess.run(self.h_data_pred, tf_dict)
    b_star = self.sess.run(self.b_data_pred, tf_dict)

    return u_star, v_star, s_star, h_star, b_star
