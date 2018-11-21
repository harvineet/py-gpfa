# EM implementation

from data_simulator import load_params

# Run EM
"""
def em(current_params, seq):
    # TODO
    params = load_params('../em_input.mat')

    # est_params, seq_train_cut, LLcut, iter_time
    return params, seq, None, None"""

def em(model, kernSDList):
	emMaxIters = 500
    tol        = 1e-8
    minVarFrac = 0.01
    verbose    = False
    freqLL     = 10 

    N = len(model.data);
    T = model.stack_attributes('T')
    yDim, xDim = model.params.C.shape
    LL         = []
    LLi        = 0
    iterTime   = []

    ycov = np.cov(model.stack_attributes('y'))
    varFloor   = minVarFrac * np.diag(ycov);

    for i in range(emMaxIters):

        # Time each iteration
        if verbose:
            print('\n')
        tic = timeit.default_timer()
    
        if np.remainder(i+1,freqLL) == 0 or i <= 1:
            getLL = True
        else:
            getLL = False

        # ====== E step ======
        if not np.isnan(LLi):
            LLold = LLi
        model, LLi = exact_inference_with_LL(model, getLL)
        LL.append(LLi)

        # ====== M step ======
        sum_Pauto = np.zeros((xDim, xDim))

        for n in range(N):
            sum_Pauto = sum_Pauto + np.sum(model.data[n].Vsm, 2) + np.matmul(model.data[n].xsm, model.data[n].xsm.T)

        Y           = model.stack_attributes('y')
        Xsm         = model.stack_attributes('xsm')
        sum_yxtrans = np.matmul(Y, Xsm.T)
        sum_xall    = np.sum(Xsm, 1)
        sum_yall    = np.sum(Y, 1)

        term = np.vstack((np.hstack((sum_Pauto, sum_xall.reshape(xDim,1))), np.hstack((sum_xall.T,np.sum(T)))))
        Cd = np.linalg.lstsq(term.T,np.hstack(( sum_yxtrans, sum_yall.reshape(yDim,1) )).T, rcond=None)
        Cd = Cd[0].T

        model.params.C = Cd[:, :xDim]
        model.params.d = Cd[:, -1]

        if model.params.RforceDiagonal:
            sum_yytrans = np.sum(Y * Y, 1)
            yd          = sum_yall * model.params.d
            term        = np.sum((sum_yxtrans - np.outer(model.params.d, sum_xall)) * model.params.C, 1)
            r           = np.square(model.params.d) + (sum_yytrans - 2*yd - term) / np.sum(T)
            # Set minimum private variance
            r               = np.maximum(varFloor, r)
            model.params.R  = np.diag(r)
        else:
            sum_yytrans = np.matmul(Y, Y.T)
            yd          = np.outer(sum_yall, model.params.d)
            term        = np.matmul((sum_yxtrans - np.outer(model.params.d, sum_xall.T)), model.params.C.T)
            R           = np.outer(model.params.d, model.params.d) + (sum_yytrans - yd - yd.T - term) / np.sum(T)
            model.params.R = (R + R.T) / 2
        
        if model.params.learnKernelParams:
       
            res = learn_GP_params(model, verbose, kernSDList)
            if model.params.cov_type == 'rbf':
                model.params.gamma = res.gamma
            elif model.params.cov_type == 'tri':
                model.params.a     = res.a
            elif model.params.cov_type == 'logexp':
                model.params.a     = res.a

        if model.params.learnGPNoise: 
            model.params.eps = res.eps
        
        tEnd = tic - timeit.default_timer()
        iterTime.append(tEnd)

            # Display the most recent likelihood that was evaluated
        if verbose:
            if getLL:
                print('       lik', LLi,'(', tEnd, 'sec iteration)\n')
            else:
                print('\n')
        else:
            if getLL:
                print('       lik', LLi, '\n')
            else:
                print('\n')

        # Verify that likelihood is growing monotonically - stop when it isn't decreasing anymore
        if i <= 2:
            LLbase = LLi
        elif LLi < LLold:
            print('\n Error: Data likelihood has decreased from', LLold,'to', LLi, '\n')
        elif (LLi-LLbase) < (1+tol)*(LLold-LLbase):
            print('\n')
            break

    if len(LL) < emMaxIters:
        printf('Fitting has converged after', len(LL), 'EM iterations.\n')

    if any(np.diag(model.params.R) == varFloor):
        print('Warning: Private variance floor used for one or more observed dimensions in GPFA.\n')

    return model, LL, iterTime
