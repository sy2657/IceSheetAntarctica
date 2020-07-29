# IceSheetAntarctica
physics-informed NN on Ice Sheet equations

Details about setup: 

In the orignial HFM cylinder flower code,
they load t_star as (T,1) 
x_star as (N,1)
y_star as (N,1) 

and they load C_star, U_star, V_star, and P_star as ( N , T) matrices.


where T is the total time and N is the flattened value N = n x n in 2 dimensions.

idx_x is the random re-ordering of values from 1 to N. 

They create t_data, x_data, y_data, and c_data from [:, idx_t][idx_x, :].flatten()[:,None].


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We test sample inputs to see what their code is doing.

We test a smaller matrix for T = 2 and N= 10,

We set them to be vectors of entries [1, ..., T] and [1, ..., N]:

t_star = np.arange(2) # T x 1 and
x_star = np.arange(10) # N x 1 and
y_star = np.arange(10) # N x 1.

and call reshape to make into the shapes of T x 1 and N x 1: 

x_star = x_star.reshape(10, 1)
y_star = y_star.reshape(10,1)
t_star = t_star.reshape(2,1)

Then we call the same code

T_star = np.tile(t_star, (1,N)).T # N x T
X_star = np.tile(x_star, (1,T))

which gives

T_star = 
array([[0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1],
       [0, 1]])

X_star = 
array([[0, 0],
       [1, 1],
       [2, 2],
       [3, 3],
       [4, 4],
       [5, 5],
       [6, 6],
       [7, 7],
       [8, 8],
       [9, 9]])

We set variables as their code:

idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_data-2, replace=False)+1, np.array([T-1])] )
idx_x = np.random.choice(N, N_data, replace=False)
       

U_star = np.arange(20) # N x T
U_star = U_star.reshape(10, 2)
C_star = U_star

U_star or C_star is equal to:

array([[ 0,  1],
       [ 2,  3],
       [ 4,  5],
       [ 6,  7],
       [ 8,  9],
       [10, 11],
       [12, 13],
       [14, 15],
       [16, 17],
       [18, 19]])

and where 
idx_t = array([0, 1])
idx_x = array([2, 7, 5, 3, 8, 0, 1, 4, 6, 9])


Then  C_star[:, idx_t][idx_x,:] is

array([[ 4,  5],
       [14, 15],
       [10, 11],
       [ 6,  7],
       [16, 17],
       [ 0,  1],
       [ 2,  3],
       [ 8,  9],
       [12, 13],
       [18, 19]])
       
We see that the [idx_t,:] part reorders the elements of each row.

We see that the [idx_x,:] part of  C_star[:, idx_t][idx_x,:] reorders the rows to fit the order [2, 7, 5, 3, 8, 0, 1, 4, 6, 9].

Given the time dimension of 1, we only apply [idx_x, :] and the command 

C_star[idx_x, :] 

has the same effect.

Now we examine what t_data and x_data look like. 

T_star[:,idx_t][idx_x,:] is the same, but elements in each row's array are switched:

array([[1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0],
       [1, 0]])

X_star[:,idx_t] is equal to X_star because the rows have the repeated elements.

x_data = X_star[:,idx_t][idx_x,:] is then equal to:
array([[2, 2],
       [7, 7],
       [5, 5],
       [3, 3],
       [8, 8],
       [0, 0],
       [1, 1],
       [4, 4],
       [6, 6],
       [9, 9]])

Flattening it turns it into a vector array([2,2, 7,7 , 5,5, ..., 9,9[) of dimension 1 by 20. 

then [:,None] flips the dimension to be (20, 1),

array([[2],
       [2],
       [7],
       [7],... ])


