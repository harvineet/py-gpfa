# Example script to run methods on sample data

# Code modified from the version by Byron Yu byronyu@stanford.edu, John Cunningham jcunnin@stanford.edu

from extract_traj import extract_traj, mean_squared_error, goodness_of_fit_rsquared
from data_simulator import load_data
import numpy as np
from core_gpfa.postprocess import postprocess
from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error
import matplotlib.pyplot as plt

# set random seed for reproducibility
# np.random.seed(1)

RUN_ID = 1
OUTPUT_DIR = './output/'+str(RUN_ID)+'/'
# INPUT_FILE = '../em_input_new.mat'
INPUT_FILE = '../fake_data2_w_genparams.mat' # '../em_input_new.mat', '../fake_data2_w_genparams.mat', '../fake_data_w_genparams.mat'

x_dim = 2 # latent dimension
method = 'gpfa'
param_cov_type = 'sm' # 'rbf', 'sm'
param_Q = 2
num_folds = 2
kern_SD = 30

# Load data
dat = load_data(INPUT_FILE)

# Extract trajectories
result = extract_traj(output_dir=OUTPUT_DIR, data=dat, method=method, x_dim=x_dim,\
                            param_cov_type=param_cov_type, param_Q = param_Q, num_folds = num_folds)

# Orthonormalize trajectories
# Returns results for the last run cross-validation fold, if enabled
(est_params, seq_train, seq_test) = postprocess(result['params'], result['seq_train'],\
                                                 result['seq_test'], method, kern_SD)

print("LL for training: %.4f, for testing: %.4f, method: %s, x_dim:%d, param_cov_type:%s"\
         % (result['LLtrain'], result['LLtest'], method, x_dim, param_cov_type))

# Output filenames for plots
output_file = OUTPUT_DIR+"/"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type

# Plot trajectories in 3D space
if x_dim >=3:
    plot_3d(seq_train, 'x_orth', dims_to_plot=[0,1,2], output_file=output_file)

# Plot each dimension of trajectory
# plot_1d(seq_train, 'x_sm', result['bin_width'], output_file=output_file)
plot_1d(seq_train, 'x_orth', result['bin_width'], output_file=output_file)

# Prediction error and extrapolation plots on test set
if len(seq_test)>0:
    # Change to 'x_orth' to get prediction error for orthogonalized trajectories
    mean_error_trials = mean_squared_error(seq_test, 'x_orth')
    print("Mean sequared error across trials: %.4f" % mean_error_trials)

    r2_trials = goodness_of_fit_rsquared(seq_test, x_dim, 'x_orth')
    print("R^2 averaged across trials: %s" % np.array_str(r2_trials, precision=4))

    # # Plot each dimension of trajectory, test data
    # plot_1d(seq_test, 'x_orth', result['bin_width'])
    # Change to 'x_orth' to plot orthogonalized trajectories
    plot_1d_error(seq_test, 'xsm', result['bin_width'], output_file=output_file)

# Plot all figures
plt.show()

# Cross-validation to find optimal state dimensionality
# TODO