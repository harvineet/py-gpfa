# Run GPFA to extract trajectories

import numpy as np
from core_gpfa.fastfa import fastfa # CHECK if import works
from core_gpfa.exact_inference_with_LL import exact_inference_with_LL
from core_gpfa.em import em
from Seq_Data_Class import Param_Class
import scipy.io as sio
from core_gpfa.init_sm_hyper import init_sm_hyper, init_sm_hyper_v2

# Skip or trim sequences to same length
def cut_trials(seq_train, seg_length=20):
    # TODO
    return seq_train

def save_results(fname, result):
    # Saving a dict with keys: 'LL', 'params', 'seq_train' and 'seq_test'
    sio.savemat(fname, mdict=result, format='5')

def gpfa_engine(seq_train, seq_test, fname, x_dim, bin_width, param_cov_type='rbf',
    param_Q = 3, start_tau=100, start_eps=1e-3, min_var_frac=0.01):
    # seq_train - array with 3 tuples of -
    #   trialId (1 x 1)   - unique trial identifier
    #   y (# neurons x T) - neural data
    #   T (1 x 1)         - number of timesteps
    # start_tau - GP timescale initialization in msec
    # start_eps - GP noise variance initialization

    # For compute efficiency, train on equal-length segments of trials
    seq_train_cut = cut_trials(seq_train) # TODO

    # Initialize state model parameters
    # Initialize GP params

    param_eps = start_eps * np.ones((x_dim,))       # GP noise variance
    kernSDList = 30
    initialize_hyperparam = True

    if param_cov_type == 'rbf':
        param_gamma = (bin_width / start_tau)**2 * np.ones((x_dim,))

    elif param_cov_type == 'sm':
        param_gamma = []
        for i in range(x_dim):
            weights = np.ones(param_Q).tolist()
            weights = weights / np.sum(weights)
            weights = weights.tolist()
            mu = np.random.uniform(0, 1, param_Q).tolist()
            vs = np.random.uniform(0, 1, param_Q).tolist()
            param_gamma.append(weights + mu + vs)

    # Initialize observation model parameters
    # Run FA to initialize parameters
    y_all = np.concatenate([trial.y for trial in seq_train_cut], 1)

    print('\nRunning FA model for initialization\n')

    fa_params_L, fa_params_Ph, fa_params_d, _ = fastfa(y_all, x_dim) # TODO Fast FA

    param_d = fa_params_d
    param_C = fa_params_L # TODO faParams.L
    param_R = np.diag(fa_params_Ph) # TODO diag(faParams.Ph)

    # Define parameter constraints
    param_notes_learnKernelParams = True
    param_notes_learnGPNoise      = False
    param_notes_RforceDiagonal    = True

    # TODO Separate params for rbf and sm
    current_params = Param_Class(param_cov_type, param_gamma, 
                                    param_eps, param_d, param_C, param_R,
                                    param_notes_learnKernelParams, param_notes_learnGPNoise,
                                    param_notes_RforceDiagonal, param_Q)

    if initialize_hyperparam and param_cov_type == 'sm':
        print('\nRunning E-step for initializing hyperparameters for SM\n')
        (seq_train, _) = exact_inference_with_LL(seq_train, current_params, getLL=False)
        init_gamma = np.zeros((len(seq_train), current_params.Q*3))
        # Calculate gamma for each latent dimension and each trial
        for d in range(x_dim):
            for i in range(len(seq_train)):
                init_train_x = np.arange(seq_train[i].T).reshape((seq_train[i].T,1))
                init_train_y = seq_train[i].xsm[d,:].T
                hyper_params = init_sm_hyper(x=init_train_x, y=init_train_y, Q=param_Q)
                # hyper_params = init_sm_hyper_v2(train_x=init_train_x, train_y=init_train_y, num_mixtures=param_Q)
                init_gamma[i, :] = hyper_params

            # Initialize with mean
            print(np.mean(init_gamma, axis=0))
            current_params.gamma[d] = np.mean(init_gamma, axis=0)
        print("initial hyper parameters", current_params.gamma)

    # Fit model parameters
    print('\nFitting GPFA model\n')
  
    (est_params, seq_train_cut, LLcut, iter_time) = em(current_params, seq_train_cut, kernSDList, min_var_frac)

    # Extract trajectories for original, unsegmented trials
    # using learned parameters
    (seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params, getLL=True)

    # Assess generalization performance
    # TODO

    LLtest = np.nan
    if len(seq_test)>0:
        (_, LLtest) = exact_inference_with_LL(seq_test, est_params, getLL=True)

    result = dict({'LLtrain':LLtrain, 'LLtest':LLtest, 'params':est_params, 'seq_train':seq_train,\
                 'seq_test':seq_test, 'bin_width':bin_width})

    # Save results
    save_results(fname, result)

    return result