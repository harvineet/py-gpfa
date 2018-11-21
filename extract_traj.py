# Extracting trajectory from given data

import os
from core_gpfa.gpfa_engine import gpfa_engine # CHECK if works

def extract_traj(output_dir, data, method='gpfa', x_dim=3):
    
    bin_width = 20 # in msec # NOT REQUIRED
    num_folds = 0
    
    # Create results directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Obtain binned spike counts
    # NOT REQUIRED, check input data format

    # Divide data into cross-validation train and test folds
    # TODO, all train data
    seq_train = data # TODO, change to numpy.array
    seq_test = [] # TODO, change to numpy.array

    # Remove inactive units based on training set
    # TODO

    # Check if training data covariance is full rank
    # TODO

    # If doing cross-validation, don't use private noise variance floor
    # TODO, set minVarFrac and pass to gpfa_engine

    # Name of results file
    output_file = output_dir+"/"+method+"_xdim_"+str(x_dim)
    
    # Call gpfa
    result = None
    result = gpfa_engine(seq_train=seq_train, seq_test=seq_test, fname=output_file,
        x_dim=x_dim, bin_width=bin_width)

    return result