modelice = HFMI( x_data, y_data, u_data, v_data, s_data, h_data,b_data,x_eqns, y_eqns, layers, batch_size)

modelice.train(total_time = 20, learning_rate=1e-3)

u_pred, v_pred, s_pred, h_pred, b_pred = modelice.predict(x_data, y_data)
