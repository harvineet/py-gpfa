# Example script to run methods on sample data

# Code modified from the version by Byron Yu byronyu@stanford.edu, John Cunningham jcunnin@stanford.edu

from extract_traj import extract_traj, mean_squared_error
from data_simulator import load_data
import numpy as np
from core_gpfa.postprocess import postprocess
from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error

# set random seed for reproducibility
np.random.seed(1)

RUN_ID = 1
OUTPUT_DIR = './output/'+str(RUN_ID)+'/'
INPUT_FILE = '../em_input_new.mat' # '../sample.mat'

x_dim = 8 # latent dimension
method = 'gpfa'
param_cov_type = 'rbf'
kern_SD = 30

# Load data
# TODO
dat = load_data(INPUT_FILE)

# Extract trajectories
result = extract_traj(output_dir=OUTPUT_DIR, data=dat, method=method, x_dim=x_dim, param_cov_type=param_cov_type)

# Orthonormalize trajectories
# Returns results for the last run cross-validation fold, if enabled
(est_params, seq_train, seq_test) = postprocess(result['params'], result['seq_train'],\
                                                 result['seq_test'], method, kern_SD)

print("LL for training: %.4f, for testing: %.4f" % (result['LLtrain'], result['LLtest']))

# Plot trajectories in 3D space
plot_3d(seq_train, 'x_orth', dims_to_plot=[0,1,2])

# Plot each dimension of trajectory
plot_1d(seq_train, 'x_orth', result['bin_width'])

# Prediction error and extrapolation plots on test set
if len(seq_test)>0:
    mean_error_trials = mean_squared_error(seq_test)
    print("Mean sequared error across trials: %.4f" % mean_error_trials)

    # # Plot each dimension of trajectory, test data
    # plot_1d(seq_test, 'x_orth', result['bin_width'])
    plot_1d_error(seq_test, 'x_orth', result['bin_width'])

# Cross-validation to find optimal state dimensionality
# TODO