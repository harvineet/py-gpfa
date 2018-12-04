# Run GPFA to extract trajectories

import numpy as np
from core_gpfa.fastfa import fastfa # CHECK if import works
from core_gpfa.exact_inference_with_LL import exact_inference_with_LL
from core_gpfa.em import em
from Seq_Data_Class import Param_Class
import scipy.io as sio

# Skip or trim sequences to same length
def cut_trials(seq_train, seg_length=20):
    # TODO
    return seq_train

def save_results(fname, result):
    # Saving a dict with keys: 'LL', 'params' and 'seq'
    sio.savemat(fname, mdict=result, format='5')

def gpfa_engine(seq_train, seq_test, fname, x_dim, bin_width,
    start_tau=100, start_eps=1e-3):
    # seq_train - array with 3 tuples of -
    #   trialId (1 x 1)   - unique trial identifier
    #   y (# neurons x T) - neural data
    #   T (1 x 1)         - number of timesteps
    # start_tau - GP timescale initialization in msec
    # start_eps - GP noise variance initialization

    # For compute efficiency, train on equal-length segments of trials
    seq_train_cut = cut_trials(seq_train) # TODO

    # Initialize state model parameters
    param_cov_type = 'rbf'
    # GP timescale
    # Assume binWidth is the time step size.
    param_gamma = (bin_width / start_tau)**2 * np.ones((x_dim,))
    # GP noise variance
    param_eps = start_eps * np.ones((x_dim,))
    
    kernSDList = 30

    # Initialize observation model parameters
    # Run FA to initialize parameters
    y_all = np.concatenate([trial.y for trial in seq_train_cut], 1)

    print('\nRunning FA model for initialization\n')

    (fa_params_L, fa_params_Ph, faLL) = fastfa(y_all, x_dim) # TODO Fast FA

    param_d = np.mean(y_all, 1, keepdims=True)
    param_C = fa_params_L # TODO faParams.L
    param_R = np.diag(fa_params_Ph) # TODO diag(faParams.Ph)

    # Define parameter constraints
    param_notes_learnKernelParams = True
    param_notes_learnGPNoise      = False
    param_notes_RforceDiagonal    = True

    current_params = Param_Class(param_cov_type, param_gamma, 
                                    param_eps, param_d, param_C, param_R,
                                    param_notes_learnKernelParams, param_notes_learnGPNoise,
                                    param_notes_RforceDiagonal)

    # Fit model parameters
    print('\nFitting GPFA model\n')
  
    (est_params, seq_train_cut, LLcut, iter_time) = em(current_params, seq_train_cut, kernSDList)

    # Extract trajectories for original, unsegmented trials
    # using learned parameters
    (seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params, getLL=True)

    result = dict({'LL':LLtrain, 'params':est_params, 'seq':seq_train})

    # Assess generalization performance
    # TODO

    # Save results
    save_results(fname, result)

    return result