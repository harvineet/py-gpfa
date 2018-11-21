# FA implementation

import numpy as np

# Run FA
def fastfa(y_all, x_dim):
    # TODO

    y_dim = y_all[0].shape[0] # q
    fa_params_L = np.random.normal(loc=0.0, scale=1.0, size=(y_dim,x_dim)) # q X p
    fa_params_Ph = np.ones((y_dim,)) # q

    return (fa_params_L, fa_params_Ph, None)