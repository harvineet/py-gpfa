# Classes to store data and parameters for the GPFA model
# Usage:
#
# seq = Seq_Data_Class(trial_id, T, seq_id, y)
# seq.y = np.zeros((10,1))
import scipy.io as sio
import numpy as np

class Model_Specs:
    def __init__(self, data=None, params=None):
        self.data = data
        self.params = params

    # Load data from mat file
    def data_from_mat(self, path_to_mat):
        mat_contents = sio.loadmat(path_to_mat, mat_dtype=True) # By default, data loaded as uint8 which gives errors
        data = []
        # Loop over each component of MATLAB struct
        for i in range(mat_contents['seq'][0].size):
            # Extract trial info
            trial_id = int(mat_contents['seq'][0][i][0][0][0])
            T = int(mat_contents['seq'][0][i][1][0][0])
            seq_id = int(mat_contents['seq'][0][i][2][0][0])
            y = mat_contents['seq'][0][i][3]
            # Store in data object and append to list
            data.append(Trial_Class(trial_id, T, seq_id, y))
        self.data = data
    
    # Helper function to stack variables across trials
    # e.g., sometimes we want vector np.array([data[0].y, data[1].y, data[2].y, ...])
    def stack_attributes(self, attribute):
        if getattr(self.data[0], attribute).size is 1:
            attributes_stacked = np.zeros(len(self.data), dtype='int16')
            for i in range(len(self.data)):
                attributes_stacked[i] = getattr(self.data[i], attribute)
        else:
            attributes_stacked = np.array([]).reshape(getattr(self.data[0], attribute).shape[0], 0)
            for i in range(len(self.data)):
                attributes_stacked = np.hstack((attributes_stacked, getattr(self.data[i], attribute)))
        return attributes_stacked
            

class Trial_Class:
    def __init__(self, trial_id, T, seq_id, y):
        self.trial_id = trial_id
        self.T = T
        self.seq_id = seq_id
        self.y = y
        self.xsm = None
        self.Vsm = None
        self.VsmGP = None

    # Function to print objects
    def __repr__(self):
        return("(Trial id: %d, T: %d, seq id: %d, y: %s)" \
                    % (self.trial_id, self.T, self.seq_id, np.array_repr(self.y)))

# This gives the user flexibility to declare parameters, or load them from elsewhere
class Param_Class():
    def __init__(self, param_cov_type=None, param_gamma=None, 
                    param_eps=None, param_d=None, param_C=None, param_R=None,
                    param_notes_learnKernelParams=None, param_notes_learnGPNoise=None,
                    param_notes_RforceDiagonal=None):
        self.cov_type = param_cov_type
        self.gamma = param_gamma
        self.eps = param_eps
        self.d = param_d
        self.C = param_C
        self.R = param_R
        self.learnKernelParams = param_notes_learnKernelParams
        self.learnGPNoise = param_notes_learnGPNoise
        self.RforceDiagonal = param_notes_RforceDiagonal
    
    # Load model parameters from a .mat file
    def params_from_mat(self, path_to_mat):
        mat_contents = sio.loadmat(path_to_mat)
        # Load individual parameters
        content = mat_contents['currentParams'][0][0]
        self.cov_type = content[0][0]
        self.gamma = content[1][0]
        self.eps = content[2][0]
        self.d = content[3].T[0]
        self.C = content[4]
        self.R = content[5]

    """def fill_params(self, param_cov_type, param_gamma, 
                                param_eps, param_d, param_C, param_R,
                                param_notes_learnKernelParams, param_notes_learnGPNoise,
                                param_notes_RforceDiagonal):
                    self.cov_type = param_cov_type
                    self.gamma = param_gamma
                    self.eps = param_eps
                    self.d = param_d
                    self.C = param_C
                    self.R = param_R
                    self.learnKernelParams = param_notes_learnKernelParams
                    self.learnGPNoise = param_notes_learnGPNoise
                    self.RforceDiagonal = param_notes_RforceDiagonal"""

    # Function to print objects
    def __repr__(self):
        return("Cov type: %s\nGamma: %s\nEps: %s\nd: %s\nC: %s\nR: %s" \
                 % (self.cov_type, np.array_repr(self.gamma), np.array_repr(self.eps),
                     np.array_repr(self.d), np.array_repr(self.C), np.array_repr(self.R)))

