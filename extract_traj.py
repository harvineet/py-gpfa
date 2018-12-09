# Extracting trajectory from given data

import os
from core_gpfa.gpfa_engine import gpfa_engine
# import copy # CHECK if required

def extract_traj(output_dir, data, method='gpfa', x_dim=3):
    
    bin_width = 20 # in msec # NOT REQUIRED
    num_folds = 1 # number of splits, keep 1 for using all train data
    
    # Create results directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtain binned spike counts
    # NOT REQUIRED, check input data format

    # Divide data into cross-validation train and test folds
    # TODO, crossvalidation folds
    N = len(data)
    f_div = np.floor(np.linspace(0, N, num_folds+1))

    for cvf in range(num_folds):
        if num_folds==1:
            print("Training on all data")    
            # seq_train = data # TODO, change to numpy.array
            # seq_test = [] # TODO, change to numpy.array
        else:
            print("Cross-validation fold %d of %d" % (cvf, num_folds))
        
        test_mask = np.zeros(N, dtype=bool)
        if num_folds > 1:
            test_mask[np.arange(f_div[cvf],f_div[cvf+1], dtype=int)] = True
        train_mask = ~test_mask

        if num_folds == 1:
            # Keep original order if using all data as training set
            tr = np.arange(0, N)
        else:
            tr = np.random.permutation(N)

        train_trial_idx = tr[train_mask]
        test_trial_idx  = tr[test_mask]
        seq_train = [data[trial_num] for trial_num in train_trial_idx] # CHECK if copy.deepcopy() required
        seq_test = [data[trial_num] for trial_num in test_trial_idx]
        print(len(seq_train), len(seq_test))
        # Remove inactive units based on training set
        # TODO

        # Check if training data covariance is full rank
        # TODO

        # If doing cross-validation, don't use private noise variance floor
        # TODO, set minVarFrac and pass to gpfa_engine

        # Name of results file
        output_file = output_dir+"/"+method+"_xdim_"+str(x_dim)
        if num_folds > 1:
            output_file += "_cv"+str(cvf)
        
        # Call gpfa
        result = None
        result = gpfa_engine(seq_train=seq_train, seq_test=seq_test, fname=output_file,
            x_dim=x_dim, bin_width=bin_width)

    return result