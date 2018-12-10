# Orthonormalize trajectories to visualize

import numpy as np
from core_gpfa.exact_inference_with_LL import exact_inference_with_LL

def orthogonalize(X, C):
    """
    X_orth: orthonormalized latent variables (x_dim x T)
    C_orth: orthonormalized loading matrix (y_dim x x_dim)
    TT: linear transform applied to latent variables (x_dim x x_dim)
    """
    x_dim = C.shape[1]

    if x_dim == 1:
        TT = np.sqrt(np.matmul(C.T, C))
        C_orth = C / TT
        X_orth = np.matmul(TT, X)
    else:
        UU, DD, Vh = np.linalg.svd(C, full_matrices=False) # TODO check thin svd
        DD = np.diag(DD)
        VV = Vh.T
        # (UU, DD, VV) = svd(C, 0)
        TT = np.matmul(DD, VV.T)
        C_orth = UU
        X_orth = np.matmul(TT, X)

    return X_orth, C_orth, TT

def segment_by_trial(seq, X, fn):
    """
    seq: data structure with timesteps T
    X: orthogonalized trajectories
    fn: name of field in Trial_Class to add orthogonalized vectors to
    """

    ctr = 0
    for n in range(len(seq)):
        T = seq[n].T
        idx = np.arange(ctr, ctr+T)
        setattr(seq[n], fn, X[:, idx]) # setting value of seq[n].fn using setattr because fn is a string

        ctr = ctr + T

    return seq

def postprocess(est_params, seq_train, seq_test, method, kern_SD):
    if method=='gpfa':
        C = est_params.C
        X  = np.concatenate([np.squeeze(np.array(trial.xsm)) for trial in seq_train], 1)
        (X_orth, C_orth, _)  = orthogonalize(X, C)
        seq_train = segment_by_trial(seq_train, X_orth, 'x_orth')

        est_params.C_orth = C_orth

        # TODO Orthonormalize seq_test
        if len(seq_test)>0:
            (seq_test, LLtest) = exact_inference_with_LL(seq_test, est_params)
            X  = np.concatenate([np.squeeze(np.array(trial.xsm)) for trial in seq_test], 1)
            (X_orth, C_orth, _)  = orthogonalize(X, C)
            seq_test = segment_by_trial(seq_test, X_orth, 'x_orth')
    else:
        pass

    return est_params, seq_train, seq_test