# Simulate data from a known GPFA process

from Seq_Data_Class import Model_Specs, Trial_Class, Param_Class
import numpy as np
import scipy
# Parameters
# TODO

# Simulate
#input how many times you want to sample
def sample_data(time):
    # TODO
    seq = [Trial_Class(i, 20, 1, np.random.rand(53,20)) for i in range(10)]
    path_to_mat = 'em_input.mat'
    # Load parameters
    params = Param_Class()
    params.params_from_mat(path_to_mat)
    # Store parameters
    model = Model_Specs(None, None)
    model.params = params
    model.data_from_mat(path_to_mat)
    #create parameter
    R = params.R
    d = params.d
    C = params.C
    sigma_n = params.eps
    T = 20

    q = C.shape[0]
    p = C.shape[1]
    #R = np.zeros((q,q))
    #np.fill_diagonal(R,np.random.uniform(0,2))
    #C = np.random.uniform(0,2,q*p).reshape(q,p)
    #d = np.random.uniform(0,2,q)
    #sigma_n = np.array([0.001]*p)
    sigma_f = np.sqrt(np.array([1]*p) - np.square(sigma_n))
    tau = np.random.uniform(1,10,p)
    
    K = []
    for i in range(p):
      K.append(np.zeros((T,T)))
    for i in range(p):
      for t1 in range(len(K[i])):
        for t2 in range (len(K[i][t1])):
          if t1 != t2:
            K[i][t1][t2] = sigma_f[i] * np.exp(-(t1-t2)**2/(2*tau[i]))
          else:
            K[i][t1][t2] = sigma_f[i] + sigma_n[i]
    #sampling once
    def sample_once():
        X = []
        for i in range (len(K)):
          X.append (np.random.multivariate_normal(np.zeros(T),K[1],1)[0])
        X = np.array(X)
        Y = []
        for i in range(len(X.T)):
          Y.append(np.random.multivariate_normal((np.dot(C,X.T[i])+d),R))
        Y = np.array(Y).T
        return Y
    seq = []
    for i in range (time):
        seq.append(sample_once())
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