# FA implementation

import numpy as np
import scipy

# Run FA
def fastfa(X, zDim):


	tol        = 1e-8
	cyc        = int(1e8)
	minVarFrac = 0.01
	verbose = False

	xDim, N = X.shape
	cX = np.cov(X,bias = 1)

	if np.linalg.matrix_rank(cX) == xDim:
	    scale = np.exp(2*np.sum(np.log(np.diag(np.linalg.cholesky(cX).T)))/xDim)
	else:
	    print('WARNING in fastfa.py: Data matrix is not full rank. SAD')
	    r     = np.linalg.matrix_rank(cX)
	    e     = np.sort(np.linalg.eig(cX)[0])[::-1]
	    scale = scipy.stats.mstats.gmean(e[:r])
	    
	L     = np.sqrt(scale/zDim) * np.random.normal(loc=0.0, scale=1.0, size=(xDim, zDim))
	Ph    = np.diag(cX)
	d     = np.mean(X, 1)

	varFloor = minVarFrac * np.diag(cX);  

	I     = np.identity(zDim);
	const = -xDim/2 * np.log(2*np.pi)
	LLi   = 0 
	LL    = []

	for i in range(cyc):
	    
	    # E-step
	    iPh  = np.diag(1/Ph)
	    iPhL = np.matmul(iPh, L)
	    MM = iPh - np.matmul(np.linalg.lstsq((I + np.matmul(L.T, iPhL)).T, iPhL.T, rcond=None)[0].T, iPhL.T)
	    beta = np.matmul(L.T, MM)

	    cX_beta = np.matmul(cX, beta.T)
	    EZZ     = I - np.matmul(beta, L) + np.matmul(beta, cX_beta)

	    # Compute log likelihood
	    LLold = LLi;    
	    ldM   = np.sum(np.log(np.diag(np.linalg.cholesky(MM).T)))
	    LLi   = N*const + N*ldM - 0.5*N*np.sum(MM * cX);

	    if verbose:
	        print('')

	    LL.append(LLi) 

	    # M-step

	    L  = np.linalg.lstsq(EZZ.T, cX_beta.T, rcond=None)[0].T
	    Ph = np.diag(cX) - np.sum(cX_beta * L, 1)
	    Ph = np.maximum(Ph, varFloor)

	    if i <= 1:
	        LLbase = LLi
	    elif LLi < LLold:
	        print('VIOLATION')
	    elif ( (LLi-LLbase) < (1+tol)*(LLold-LLbase) ):
	        break

	        
	if np.any(Ph == varFloor):
	    print('Warning: Private variance floor used for one or more observed dimensions in FA.\n')
	    
	return L, Ph, d, LL

