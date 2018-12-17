from core_gpfa.postprocess import orthogonalize
from core_gpfa.make_K_big import make_K_big
from core_gpfa.util import invPerSymm, fillPerSymm
import scipy

# Performs leave-neuron-out prediction for GPFA.  This version takes 
# advantage of R being diagonal for computational savings.

# Author: Byron Yu 2009
# Translated to Python by Roman Huszar

import numpy as np
import scipy
from core_gpfa.postprocess import orthogonalize
from core_gpfa.make_K_big import make_K_big
from core_gpfa.util import invPerSymm, fillPerSymm

def cosmoother_gpfa_viaOrth_fast(seq, params, mList):

    yDim, xDim = params.C.shape
    Rinv       = np.diag(1 / np.diag(params.R))
    CRinv      = np.matmul(params.C.T, Rinv)
    CRinvC     = np.matmul(CRinv, params.C)
    blah, Corth, TT = orthogonalize(np.zeros(xDim), params.C)

    Tall = np.array([trial.T for trial in seq])
    Tu   = np.unique(Tall)

    out_seq = []
    for i in range(len(seq)):
        out_seq.append( {'dim'+str(key): np.empty((yDim, seq[key].T)) * np.nan for key in mList} )

    for j in range(Tu.size):

        T     = Tu[j];
        Thalf = int(np.ceil(T/2))

        K_big, K_big_inv, logdet_K_big = make_K_big(params, T)
        K_big = scipy.sparse.csr_matrix(K_big)

        blah = [CRinvC for _ in range(T)]

        off_diag_sparse = True
        invM, logdet_M = invPerSymm(K_big_inv + scipy.linalg.block_diag(*blah), xDim, off_diag_sparse)

        # Process all trials with length T
        nList = np.where(Tall == T)[0]
        dif = np.concatenate([trial.y for trial in seq if trial.T == T],1) - params.d.reshape((params.d.size, 1))
        CRinv_dif = np.matmul(CRinv, dif)

        for i in range(yDim):

            ci_invM    = np.zeros((Thalf, xDim*T)) * np.nan
            ci_invM_ci = np.zeros((Thalf, T)) * np.nan
            idx        = np.arange(0, xDim*T+1, xDim)
            ci         = params.C[i,:] / np.sqrt(params.R[i,i])

            for t in range(Thalf):
                bIdx         = np.arange(idx[t], idx[t+1])
                ci_invM[t,:] = np.matmul(ci, invM[bIdx,:])
            for t in range(T):
                bIdx         = np.arange(idx[t], idx[t+1])
                ci_invM_ci[:,t] = np.matmul(ci_invM[:,bIdx], ci)

            ci_invM = fillPerSymm(ci_invM, xDim, T, 1)
            term = np.linalg.lstsq( (fillPerSymm(ci_invM_ci, 1, T) - np.identity(T)) , ci_invM, rcond=None)[0]

            invM_mi = invM - np.matmul(ci_invM.T, term)

            # Subtract out contribution of neuron i 
            CRinvC_mi = CRinvC - np.outer(ci, ci.T)
            term1Mat = np.reshape(CRinv_dif - np.outer(params.C[i,:] / params.R[i,i], dif[i,:]), (xDim*T, -1),order='F')

            blkProd = np.zeros((xDim*Thalf, xDim*T))
            idx     = np.arange(0, xDim*Thalf + 1, xDim)
            for t in range(Thalf):
                bIdx            = np.arange(idx[t], idx[t+1]);
                blkProd[bIdx,:] = np.matmul(CRinvC_mi, invM_mi[bIdx,:])

            blkProd = K_big[np.arange(xDim*Thalf), :].dot(fillPerSymm(scipy.sparse.eye(xDim*Thalf, xDim*T) - blkProd, xDim, T))
            xsmMat = np.matmul( fillPerSymm(blkProd, xDim, T), term1Mat)

            ctr = 0
            for n in nList:
                xorth = np.matmul( TT, np.reshape(xsmMat[:,ctr], (xDim, T), order='F') )

                for m in mList:
                    out_seq[n]['dim'+str(m)][i,:] = np.matmul(Corth[i, np.arange(m+1)], xorth[np.arange(m+1),:]) + params.d[i]

                ctr = ctr + 1

        print('Cross-validation complete for',j+1, 'of', Tu.size, 'trial lengths\n')

    return out_seq