# Construct full GP covariance matrix
from core_gpfa.util import invToeplitz
import numpy as np

def make_K_big(params, T):
    
    x_dim = params.C.shape[1]

    idx = np.arange(0, x_dim*(T-1)+1, x_dim) # TODO check if 0 : xDim : (xDim*(T-1))

    K_big = np.zeros((x_dim*T,x_dim*T))
    K_big_inv = np.zeros((x_dim*T,x_dim*T))

    Tdif = np.tile(np.arange(1,T+1).reshape((T,1)), (1, T)) - np.tile(np.arange(1,T+1), (T, 1))
    logdet_K_big = 0

    for i in range(x_dim):
        if params.cov_type == 'rbf':
            K = (1 - params.eps[i]) \
                * np.exp(-params.gamma[i] / 2 * Tdif**2) \
                + params.eps[i] * np.eye(T)

        else: # TODO 'tri', 'logexp'
            pass

        K_big[np.ix_(idx+i, idx+i)] = K
        inv_K, logdet_K = invToeplitz(K) # TODO Trench method
        K_big_inv[np.ix_(idx+i, idx+i)] = inv_K

        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big
