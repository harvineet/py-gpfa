# Extracting trajectory from given data

import os
from core_gpfa.gpfa_engine import gpfa_engine
import numpy as np
# import copy # CHECK if required

# Mean squared error between actual and predicted latent trajectories
def mean_squared_error(seq):
    error_trials = np.zeros(len(seq))
    for n in range(len(seq)):
        x_dim = (seq[n].xsm).shape[0]
        T = seq[n].T
        # Frobenius norm
        error = np.sum(np.power(seq[n].xsm - seq[n].x, 2))
        # Normalize by x_dim*T
        error = error * 1.0 / (x_dim * T)
        error_trials[n] = error
    print("error_trials", error_trials)
    mean_error_trials = np.mean(error_trials)

    return mean_error_trials

def extract_traj(output_dir, data, method='gpfa', x_dim=3, param_cov_type='rbf', param_Q = 3, num_folds = 0):
    # num_folds: number of splits (>= 2), set 0 for using all train data

    bin_width = 20 # in msec # NOT REQUIRED
    min_var_frac = 0.01 # used in em

    # Create results directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtain binned spike counts
    # NOT REQUIRED, check input data format

    # Divide data into cross-validation train and test folds
    N = len(data)
    f_div = np.floor(np.linspace(0, N, num_folds+1))

    for cvf in range(num_folds+1):
        # TODO cvf=1 runs on all data as testing set
        # cvf=0 runs on all data as training set
        if cvf==0:
            print("Training on all data\n")
        else:
            print("Cross-validation fold %d of %d\n" % (cvf, num_folds))
        
        test_mask = np.zeros(N, dtype=bool)
        if cvf > 0:
            test_mask[np.arange(f_div[cvf-1],f_div[cvf], dtype=int)] = True
        train_mask = ~test_mask

        if cvf == 0:
            # Keep original order if using all data as training set
            tr = np.arange(0, N)
        else:
            tr = np.random.permutation(N)

        train_trial_idx = tr[train_mask]
        test_trial_idx  = tr[test_mask]
        seq_train = [data[trial_num] for trial_num in train_trial_idx] # CHECK if copy.deepcopy() required
        seq_test = [data[trial_num] for trial_num in test_trial_idx]

        # Remove inactive units based on training set
        # TODO

        # Check if training data covariance is full rank
        # TODO

        print('Number of trials in training: %d\n' % len(seq_train));
        print('Number of trials in testing: %d\n' % len(seq_test));
        print('Dimensionality of latent space: %d' % x_dim);

        if len(seq_train)==0:
            print("No examples in training set. Exiting from current cross-validation run")
            continue

        # If doing cross-validation, don't use private noise variance floor
        # TODO, set minVarFrac and pass to gpfa_engine
        if cvf > 0:
            min_var_frac = -np.inf

        # Name of results file
        output_file = output_dir+"/"+method+"_xdim_"+str(x_dim)
        if cvf > 0:
            output_file += "_cv"+str(cvf)
        
        # Call gpfa
        result = None
        result = gpfa_engine(seq_train=seq_train, seq_test=seq_test, fname=output_file,
            x_dim=x_dim, bin_width=bin_width, param_cov_type=param_cov_type, param_Q = param_Q, min_var_frac=min_var_frac)

    # Returns result of the last run fold
    return result