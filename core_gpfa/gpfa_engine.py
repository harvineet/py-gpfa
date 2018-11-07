# Run GPFA to extract trajectories

import numpy as np
from fastfa import fastfa # CHECK if import works
from exact_inference_with_LL import exact_inference_with_LL
from em import em

# Skip or trim sequences to same length
def cut_trials(seq_train, seg_length=20):
	# TODO
	return seq_train

def gpfa_engine(seq_train, seq_test, fname, x_dim, bin_width,
	start_tau=100, start_eps=1e-3):
	# seq_train - array with 3 tuples of -
	# 	trialId (1 x 1)   - unique trial identifier
	# 	y (# neurons x T) - neural data
	# 	T (1 x 1)         - number of timesteps
	# start_tau - GP timescale initialization in msec
	# start_eps - GP noise variance initialization

	# For compute efficiency, train on equal-length segments of trials
	seq_train_cut = cut_trials(seq_train)

	# Initialize state model parameters
	param_cov_type = 'rbf'
    # GP timescale
  	# Assume binWidth is the time step size.
  	param_gamma = (bin_width / start_tau)^2 * np.ones((1, x_dim))
 	# GP noise variance
  	param_eps = start_eps * np.ones((1, x_dim))

	# Initialize observation model parameters
	# Run FA to initialize parameters
	y_all = [] # TODO seqTrainCut.y

	print('\nRunning FA model for initialization\n')

	(fa_params, faLL) = fastfa(y_all, x_dim)

	param_d = np.mean(y_all, 2)
	param_C = [] # TODO faParams.L
	param_R = [] # TODO diag(faParams.Ph)

	# Define parameter constraints
	param_notes_learnKernelParams = True
	param_notes_learnGPNoise      = False
	param_notes_RforceDiagonal    = True

	current_params = [param_cov_type, param_gamma, param_eps,
					param_d, param_C, param_R,
					param_notes_learnKernelParams, param_notes_learnGPNoise,param_notes_RforceDiagonal]

	# Fit model parameters
	# TODO
	print('\nFitting GPFA model\n')
  
	(est_params, seq_train_cut, LLcut, iter_time) = em(*current_params, seqTrainCut)

	# Extract trajectories for original, unsegmented trials
	# using learned parameters
	(seq_train, LLtrain) = exact_inference_with_LL(seq_train, est_params)

	result = None

	# Assess generalization performance
	# TODO

	# Save results
	# TODO

	return result
	