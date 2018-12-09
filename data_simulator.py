# Simulate data from a known GPFA process

from Seq_Data_Class import Model_Specs, Trial_Class, Param_Class
import numpy as np
import scipy
# Parameters
# TODO

# Simulate
#input how many times you want to sample
def sample_data(kernel,params,time):
    #kernel: RBF or SM
    #params: type params_class()
    #time: how many independent trail class
    
    #create a seq of tria_class
    seq = [Trial_Class(i, 20, 1, None,None) for i in range(time)]
    
    # Load parameters
    R = params.R
    d = params.d
    C = params.C
    sigma_n = params.eps
    T = 20
    q = C.shape[0]
    p = C.shape[1]
    
    #create kernel K
    if kernel == 'RBF':
        K = []
        Tdif = np.tile(np.arange(1,T+1).reshape((T,1)), (1, T)) - np.tile(np.arange(1,T+1), (T, 1))
        diffSq = Tdif ** 2
        for i in range (time):
            const = params.eps[1]
            p = params.gamma[1] 
            temp= (1-const) * np.exp(-p / 2 * diffSq)
            Kmax = temp + const * np.identity(diffSq.shape[0])
            K.append(Kmax)
        
    if kernel == 'SM'
        K = []
        Tdif = np.tile(np.arange(1,T+1).reshape((T,1)), (1, T)) - np.tile(np.arange(1,T+1), (T, 1))
        diffSq = Tdif ** 2
        for i in range(time):
            w = params.gamma[i][:params.Q]
            m = params.gamma[i][params.Q:params.Q*2]
            v = params.gamma[i][params.Q*2:params.Q*3]
            Km = np.zeros(diffSq.shape);
            for j in range(len(w)):
                Km = Km + w[j] * np.exp(-2 * np.pi**2 * v[j]**2 * diffSq) * np.cos(2 * np.pi *  Tdif.T * m[j]) 
            K.append(Km)
            
    #sampling once
    def sample_once():
        X = []
        for i in range (len(K)):
            X.append (np.random.multivariate_normal(np.zeros(T),K[i],1)[0])
        X = np.array(X)
        Y = []
        for i in range(len(X.T)):
            Y.append(np.random.multivariate_normal((np.dot(C,X.T[i])+d),R))
        Y = np.array(Y).T
        return X,Y
    
    #adding X,Y to trial in the seq    
    for i in range (time):
        seq[i].x, seq[i].y = sample_once()
    return seq

def sample_sm(T, ntrials, xDim, Q):
    # N       - number of timesteps per trial
    # ntrials - total number of trials to generate
    # xDim    - dimensionality of latent space
    # Q       - number of SM mixtures
    
    # Preload parameters from the mat file
    path_to_mat = '/Users/romanhuszar/Documents/TimeSeries/project/workspace/em_input.mat'
    params = Param_Class()
    params.params_from_mat(path_to_mat)
    params.Q = Q
    param_cov_type = 'sm'
    # Now we generate parameters of each GP
    gamma = []
    for i in range(xDim):
        weights = np.random.uniform(0, 1, params.Q).tolist()
        weights = weights / np.sum(weights)
        weights = weights.tolist()
        mu = np.random.uniform(0, 1, params.Q).tolist()
        vs = np.random.uniform(0, 1, params.Q).tolist()
        gamma.append(weights + mu + vs)
    params.gamma = gamma
    
    # Generate covariance matrix for each latent dimension
    K = []
    Tdif = np.tile(np.arange(1,T+1).reshape((T,1)), (1, T)) - np.tile(np.arange(1,T+1), (T, 1))
    diffSq = Tdif ** 2
    for i in range(xDim):
        w = params.gamma[i][:params.Q]
        m = params.gamma[i][params.Q:params.Q*2]
        v = params.gamma[i][params.Q*2:params.Q*3]
        Km = np.zeros(diffSq.shape);
        for j in range(len(w)):
            Km = Km + w[j] * np.exp(-2 * np.pi**2 * v[j]**2 * diffSq) * np.cos(2 * np.pi *  Tdif.T * m[j]) 
        K.append(Km)
    
    seq = []
    for i in range(ntrials):
        Y = generate_trial_data(params, K, xDim, T)
        seq.append( Trial_Class(i, T, i, Y) )
    
    return params, seq


def generate_trial_data(params, K, xDim, T):
    X = np.zeros((xDim, T))
    for i in range(xDim):
        X[i,:] = np.random.multivariate_normal( np.zeros(T), K[i] )
    Y = np.zeros((params.C.shape[0], T))
    for i in range(T):  
        Y[:,i] = np.random.multivariate_normal( np.matmul(params.C, X[:,i]) + params.d, params.R)
    return Y

# Save to file
def save_data(filepath,sample_data):
    # TODO
    save = {}
    
    for i in range(len(sample_data)):
        save[str(i)] = sample_data[i]
    scipy.io.savemat(filepath,save,do_compression=True)
    print("Saved file at", filepath)

def save_params(filepath,params):
    save = {'currentParams':[]}
    save['currentParams'].append([[params.cov_type]])
    save['currentParams'].append([[params.gamma]])
    save['currentParams'].append([params.eps])
    save['currentParams'].append(np.array([params.d]).T)
    save['currentParams'].append(params.C)
    save['currentParams'].append(params.R)
    scipy.io.savemat(filepath,save,do_compression=True)
    print("Saved file at", filepath)
    
# Load from file
def load_data(filepath):
    # TODO
    model_data = Model_Specs()
    model_data.data_from_mat(filepath)
    data = model_data.data

    return data

# Load parameters from mat file
def load_params(filepath):
    params = Param_Class()
    params.params_from_mat(filepath)

    return params

if __name__ == "__main__":
    print("Simulating data")
    sample_data = sample_data(20)
    print(sample_data)
    save_data('sample.mat',sample_data)