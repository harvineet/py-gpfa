# Example script to run methods on sample data

# Code modified from the version by Byron Yu byronyu@stanford.edu, John Cunningham jcunnin@stanford.edu

from extract_traj import extract_traj
from data_simulator import load_data
import numpy as np
from core_gpfa.postprocess import postprocess
from core_gpfa.plot_3d import plot_3d, plot_1d

# set random seed for reproducibility
np.random.seed(1)

RUN_ID = 1
OUTPUT_DIR = './output/'+str(RUN_ID)+'/'
INPUT_FILE = '../em_input_new.mat' # '../sample.mat'

x_dim = 8 # latent dimension
method = 'gpfa'
kern_SD = 30

# Load data
# TODO
dat = load_data(INPUT_FILE)

# Extract trajectories
result = extract_traj(output_dir=OUTPUT_DIR, data=dat, method=method, x_dim=x_dim)

# Orthonormalize trajectories
(est_params, seq_train) = postprocess(result, method, kern_SD)

# Plot trajectories in 3D space
plot_3d(seq_train, 'x_orth', dims_to_plot=[0,1,2])

# TODO plots for each dimension of trajectory
plot_1d(seq_train, 'x_orth', result['bin_width'])

# Cross-validation to find optimal state dimensionality
# TODO