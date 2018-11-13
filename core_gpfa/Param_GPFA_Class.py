# Class to store parameters while estimating GPFA model
# Usage:
#
# current_params = Param_GPFA_Class(param_cov_type, param_gamma, param_eps,
#                   param_d, param_C, param_R,
#                   param_notes_learnKernelParams, param_notes_learnGPNoise,param_notes_RforceDiagonal)
# current_params.param_cov_type = 'rbf'

class Param_GPFA_Class():
    def __init__(self, param_cov_type, param_gamma, 
                    param_eps, param_d, param_C, param_R,
                    param_notes_learnKernelParams, param_notes_learnGPNoise,
                    param_notes_RforceDiagonal):
        self.param_cov_type = param_cov_type
        self.param_gamma = param_gamma
        self.param_eps = param_eps
        self.param_d = param_d
        self.param_C = param_C
        self.param_R = param_R
        self.param_notes_learnKernelParams = param_notes_learnKernelParams
        self.param_notes_learnGPNoise = param_notes_learnGPNoise
        self.param_notes_RforceDiagonal = param_notes_RforceDiagonal