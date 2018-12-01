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