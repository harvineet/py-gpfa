# Classes to store data and parameters for the GPFA model
# Usage:
#
# seq = Seq_Data_Class(trial_id, T, seq_id, y)
# seq.y = np.zeros((10,1))
import scipy.io as sio

class Model_Specs:
    def __init__(self, data, params):
        self.data = data
        self.params = params

    # Load data from mat file
    def data_from_mat(self, path_to_mat):
        mat_contents = sio.loadmat(path_to_mat)
        data = []
        # Loop over each component of MATLAB struct
        for i in range(mat_contents['seq'][0].size):
            # Extract trial info
            trial_id = mat_contents['seq'][0][i][0][0][0]
            T = mat_contents['seq'][0][i][1][0][0]
            seq_id = mat_contents['seq'][0][i][2][0][0]
            y = mat_contents['seq'][0][i][3]
            # Store in data object and append to list
            data.append(Trial_Class(trial_id, T, seq_id, y))
        self.data = data

class Trial_Class:
    def __init__(self, trial_id, T, seq_id, y):
        self.trial_id = trial_id
        self.T = T
        self.seq_id = seq_id
        self.y = y
    # Function to print objects
    def __repr__(self):
        return "("+",".join(map(str, [self.trial_id, self.T, self.seq_id] + list(self.y.shape)))+")"

# This gives the user flexibility to declare parameters, or load them from elsewhere
class Param_Class():
    def __init__(self):
        self.cov_type = None
        self.gamma = None
        self.eps = None
        self.d = None
        self.C = None
        self.R = None
        self.learnKernelParams = None
        self.learnGPNoise = None
        self.RforceDiagonal = None
    
    # Load model parameters from a .mat file
    def params_from_mat(self, path_to_mat):
        mat_contents = sio.loadmat(path_to_mat)
        # Load individual parameters
        self.cov_type = mat_contents['currentParams'][0][0][0][0]
        self.gamma = mat_contents['currentParams'][0][0][1][0]
        self.eps = mat_contents['currentParams'][0][0][2][0]
        self.d = mat_contents['currentParams'][0][0][3].T[0]
        self.C = mat_contents['currentParams'][0][0][4]
        self.R = mat_contents['currentParams'][0][0][5]

    def fill_params(self, param_cov_type, param_gamma, 
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
        self.RforceDiagonal = param_notes_RforceDiagonal

