
import numpy as np
from core_gpfa.util import invToeplitz
from scipy.optimize import minimize

def learn_GP_params(seq, current_params):
    # Learn GP hyperparameters for each latent dimension separately
    # This code uses the BFGS optimizer which estimates the gradient
    
    isVerbose = False

    oldParams = current_params.gamma.copy()
    xDim    = len(oldParams)
    out_params = []

    if current_params.cov_type == 'rbf':
        fname     = grad_rbf
    elif current_params.cov_type == 'sm':
        fname = grad_sm
        

    precomp = makePrecomp(seq, xDim)

    for i in range(xDim):

        initp = np.log(oldParams[i])           # Single param per latent dim

        curr_args = {
            'Tall': precomp['Tall'][i],
            'T': precomp['T'][i], 
            'Tdif': precomp['Tdif'], 
            'difSq': precomp['difSq'][i], 
            'numTrials': precomp['numTrials'][i],
            'PautoSUM': precomp['PautoSUM'][i]
        }
        if current_params.cov_type == 'rbf':
            
            const = current_params.eps[i]
            res = minimize(fun = fname, 
                           x0 = initp, 
                           args = (curr_args, const), 
                           method='BFGS',
                           options={'disp': isVerbose}) 
            out_params.append(np.exp(res.x[0]))
        
        elif current_params.cov_type == 'sm':
            Q = current_params.Q
            # Weights must sum to 1
            wbound = tuple((-10, None) for _ in range(Q))
            gaussbound = tuple((None, None) for _ in range(Q*2))
            bnds = wbound + gaussbound
            res = minimize(fun = fname , 
                           x0 = initp , 
                           args = (curr_args, Q), 
                           method='L-BFGS-B',
                           bounds=bnds ,
                           options={'disp': isVerbose}), 
            res = np.exp(res[0].x)
            res[:Q] = res[:Q] / np.sum(res[:Q])
            out_params.append(res.tolist())
    
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

def grad_rbf(p, curr_args, const):
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

def grad_sm(p, curr_args, Q):
    # Cost function for spectral mixture kernel
    # No gradient is returned

    p = np.exp(p).tolist()
    # Force weights to sum to 1
    # w = (p[:Q] / np.sum(p[:Q])).tolist()
    w = p[:Q]
    m = p[Q:Q*2]
    v = p[Q*2:Q*3]
    # Generate the covariance for given setting of parameters
    Kmax = np.zeros(curr_args['difSq'].shape)
    for i in range(Q):
        Kmax = Kmax + w[i] * np.exp(-2 * np.pi**2 * v[i]**2 * curr_args['difSq']) * np.cos(2 * np.pi *  curr_args['Tdif'].T * m[i]) 

    T = curr_args['T'][0]
    Kinv, logdet_K = invToeplitz(Kmax[0:T, 0:T])

    f = -(- 0.5 * curr_args['numTrials'] * logdet_K - 0.5 * np.dot(curr_args['PautoSUM'].flatten(), Kinv.flatten()))

    return f