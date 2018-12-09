# Construct full GP covariance matrix
from core_gpfa.util import invToeplitz
import numpy as np

def make_K_big(params, T):
    
    x_dim = params.C.shape[1]

    idx = np.arange(0, x_dim*(T-1)+1, x_dim) # TODO check if 0 : xDim : (xDim*(T-1))

    K_big = np.zeros((x_dim*T,x_dim*T))
    K_big_inv = np.zeros((x_dim*T,x_dim*T))

    Tdif = np.tile(np.arange(1,T+1).reshape((T,1)), (1, T)) - np.tile(np.arange(1,T+1), (T, 1))
    diffSq = Tdif ** 2
    logdet_K_big = 0

    for i in range(x_dim):
        if params.cov_type == 'rbf':
            K = (1 - params.eps[i]) \
                * np.exp(-params.gamma[i] / 2 * diffSq) \
                + params.eps[i] * np.eye(T)

        elif params.cov_type == 'sm': # TODO 'tri', 'logexp'
            w = params.gamma[i][:params.Q]
            m = params.gamma[i][params.Q:params.Q*2]
            v = params.gamma[i][params.Q*2:params.Q*3]
            K = np.zeros(diffSq.shape);
            for j in range(len(w)):
                K = K + w[j] * np.exp(-2 * np.pi**2 * v[j]**2 * diffSq) * np.cos(2 * np.pi *  Tdif.T * m[j]) 

        K_big[np.ix_(idx+i, idx+i)] = K
        inv_K, logdet_K = invToeplitz(K) # TODO Trench method
        K_big_inv[np.ix_(idx+i, idx+i)] = inv_K

        logdet_K_big = logdet_K_big + logdet_K

    return K_big, K_big_inv, logdet_K_big
