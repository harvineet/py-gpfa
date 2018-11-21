# Extracts latent trajectories given GPFA model parameters

import numpy as np
from scipy import sparse
import scipy.linalg
from core_gpfa.util import invPerSymm, fillPerSymm, logdet
from core_gpfa.make_K_big import make_K_big

def exact_inference_with_LL(seq, params, getLL = False):

    y_dim, x_dim = params.C.shape

    # Precomputations
    if params.RforceDiagonal:
        R_inv = np.diag(1./np.diag(params.R))
        logdet_R = np.sum(np.log(np.diag(params.R)))
    else:
        R_inv = np.linalg.inv(params.R)
        R_inv = (R_inv+R_inv.T) / 2
        logdet_R = logdet(params.R)

    CRinv  = np.matmul(params.C.T, R_inv)
    CRinvC = np.matmul(CRinv, params.C)

    T_all = [s.T for s in seq]
    Tu   = np.unique(T_all)
    LL   = 0.0

    # Process: 1. for each unique trial length, 2. select trials with that length, 
    # 3. perform inference and log-likelihood computation
    for j in range(len(Tu)):
        T = Tu[j]

        K_big, K_big_inv, logdet_K_big = make_K_big(params, T)

        K_big = sparse.csr_matrix(K_big) # TODO check other sparse formats

        blah = [CRinvC for _ in range(T)]

        off_diag_sparse = True
        invM, logdet_M = invPerSymm(K_big_inv + scipy.linalg.block_diag(*blah), x_dim, off_diag_sparse)

        # compute posterior covariance matrix
        Vsm = np.full((x_dim, x_dim, T), np.nan)
        idx = np.arange(0, x_dim*(T)+1, x_dim) # TODO check if 1: x_dim : (x_dim*T + 1)

        for t in range(T):
            cIdx = np.arange(idx[t],idx[t+1])
            Vsm[:,:,t] = invM[cIdx, cIdx]

        # T x T posterior covariance for each GP
        VsmGP = np.full((T, T, x_dim), np.nan)
        idx   = np.arange(0, x_dim*(T-1)+1, x_dim) # TODO check if 0 : x_dim : (x_dim*(T-1)) 
        # index offset, so no change for 0-indexing

        for i in range(x_dim):
            VsmGP[:,:,i] = invM[idx+i,idx+i]

        # Process all trials with length T
        n_list = [i for i,x in enumerate(T_all) if x == T]
        dif = np.concatenate([trial.y for trial in seq if trial.T == T], 1) \
                - params.d.reshape((params.d.shape[0],1)) # y_dim x sum(T)
        term1Mat = np.matmul(CRinv, dif).reshape((x_dim*T, -1), order="F").copy() # (x_dim*T) x length(n_list) 
        # Fortran-like order for reshaping

        # Compute blk_prod = CRinvC_big * invM efficiently
        # blk_prod is block persymmetric, so just compute top half
        T_half = int(np.ceil(T/2.))
        blk_prod = np.zeros((x_dim*T_half, x_dim*T))
        idx = np.arange(0, x_dim*T_half+1, x_dim) # TODO check if 1: x_dim : (x_dim*T_half + 1)

        for t in range(T_half):
            bIdx = np.arange(idx[t],idx[t+1])
            blk_prod[bIdx,:] = np.matmul(CRinvC, invM[bIdx,:])

        # print((sparse.eye(m=x_dim*T_half, n=x_dim*T) - blk_prod).shape)
        blk_prod = sparse.csr_matrix.dot(K_big[0:(x_dim*T_half), :],
                    fillPerSymm(sparse.eye(m=x_dim*T_half, n=x_dim*T) - blk_prod, x_dim, T))
        # print(fillPerSymm(sparse.eye(m=x_dim*T_half, n=x_dim*T) - blk_prod, x_dim, T))
        # print(fillPerSymm(blk_prod, x_dim, T).shape)
        xsmMat = sparse.csr_matrix.dot(fillPerSymm(blk_prod, x_dim, T), term1Mat)

        ctr = 0
        for n in n_list:
            seq[n].xsm = xsmMat[:,ctr].reshape((x_dim, T)) # check if changing inplace or copy needed
            seq[n].Vsm = Vsm
            seq[n].VsmGP = VsmGP

            ctr += 1
        # Compute data likelihood
        if getLL:
            val = -T * logdet_R - logdet_K_big - logdet_M - y_dim * T * np.log(2*np.pi)
            LL  = LL + len(n_list) * val - np.sum(np.matmul(R_inv, dif) * dif) \
                    + np.sum(np.matmul(term1Mat.T, invM) * term1Mat.T)

    if getLL:
        LL = LL / 2.
    else:
        LL = np.nan

    return seq, LL