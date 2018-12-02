
import numpy as np
from core_gpfa.util import invToeplitz
from scipy.optimize import minimize

def learn_GP_params(seq, current_params):
    # Learn GP hyperparameters for each latent dimension separately
    # This code uses the BFGS optimizer which estimates the gradient
    
    isVerbose = False

    if current_params.cov_type == 'rbf':
        oldParams = current_params.gamma.copy()
        fname     = grad_betgam
        xDim    = oldParams.size
        out_params = np.zeros(xDim)

    precomp = makePrecomp(seq, xDim)

    for i in range(xDim):

        const = current_params.eps[i]

        if current_params.cov_type == 'rbf':
            initp = np.log(oldParams[i])           # Single param per latent dim
        elif current_params.cov_type == 'sm':
            initp = np.log(oldParams[:,i])         # More than one param per latent dim

        curr_args = {
            'Tall': precomp['Tall'][i],
            'T': precomp['T'][i], 
            'Tdif': precomp['Tdif'], 
            'difSq': precomp['difSq'][i], 
            'numTrials': precomp['numTrials'][i],
            'PautoSUM': precomp['PautoSUM'][i]
        }

        res = minimize(fun = fname, 
                       x0 = initp, 
                       args = (curr_args, const), 
                       method='BFGS',
                       options={'disp': isVerbose}) 

        if current_params.cov_type == 'rbf':
            out_params[i] = np.exp(res.x)
    
    return out_params
 
def makePrecomp( seq , xDim ):

    Tall = np.array([trial.T for trial in seq])
    Tmax = np.max(Tall)
    Tdif = np.tile(np.arange(1,Tmax+1,1).reshape(Tmax,1), (1, Tmax)) - np.tile(np.arange(1,Tmax+1,1),(Tmax,1))

    absDif = []
    difSq = []
    Talll = []

    for i in range(xDim):
        absDif.append(abs(Tdif)) 
        difSq.append(np.square(Tdif))
        Talll.append(Tall)
    
    # This is assumed to be unique - Tu is a scalar
    # the code won't work if Tu is a vector
    Tu = np.unique(Talll)   
    nList = []
    T = []
    numTrials = []
    PautoSUM = []

    #  Loop once for each state dimension (each GP)
    for i in range(xDim):
        nList.append(np.where(Tall == Tu)[0])
        T.append(Tu)
        numTrials.append(nList[i].size)
        PautoSUM.append(np.zeros((Tu[0], Tu[0])))
 
    # Loop once for each dimension
    for i in range(xDim):
        # Loop once for each trial in dimension
        for j in nList[i]:
            PautoSUM[i] = PautoSUM[i] + seq[j].VsmGP[:,:,i] + np.outer(seq[j].xsm[i,:].T, seq[j].xsm[i,:])
    
    precomp = {
        'T':T, 
        'Tall': Talll,
        'Tdif':Tdif, 
        'difSq':difSq, 
        'numTrials':numTrials,
        'PautoSUM':PautoSUM}

    return precomp

def grad_betgam(p, curr_args, const):
    # Cost function for squared exponential function
    # No gradient is returned

    Tall = curr_args['Tall']  
    Tmax = np.max(Tall)    
    temp         = (1-const) * np.exp(-np.exp(p) / 2 * curr_args['difSq'])
    Kmax         = temp + const * np.identity(Tmax)

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 * np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f