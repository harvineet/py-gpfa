import numpy as np
import numbers
import scipy.spatial.distance as spdist

# Copied from https://python-future.org/_modules/future/utils.html#old_div
def old_div(a, b):
    """
    DEPRECATED: import ``old_div`` from ``past.utils`` instead.

    Equivalent to ``a / b`` on Python 2 without ``from __future__ import
    division``.

    TODO: generalize this to other objects (like arrays etc.)
    """
    if isinstance(a, numbers.Integral) and isinstance(b, numbers.Integral):
        return a // b
    else:
        return a / b

# Taken from PyGPs source code.
# Written by Marion Neumann, Daniel Marthaler, Shan Huang, Kristian Kersting
# Source: https://github.com/marionmari/pyGPs/blob/master/pyGPs/Core/cov.py
def init_sm_hyper(x, y, Q):
    """
    Initialize hyperparameters for the spectral-mixture kernel. Weights are
    all set to be uniformly distributed, means are given by a random sample
    from a uniform distribution scaled by the Nyquist frequency, and variances are given by a random sample from a uniform distribution scaled by the max distance.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    (n, D) = x.shape
    w = np.zeros(Q)
    m = np.zeros((D, Q))
    s = np.zeros((D, Q))
    w[:] = old_div(np.std(y), Q)
    hypinit = np.zeros(Q + 2 * D * Q)

    for i in range(D):
        # Calculate distances
        xslice = np.atleast_2d(x[:, i]).T
        d2 = spdist.cdist(xslice, xslice, 'sqeuclidean')
        if n > 1:
            d2[d2 == 0] = d2[0, 1]
        else:
            d2[d2 == 0] = 1
        minshift = np.min(np.min(np.sqrt(d2)))
        nyquist = old_div(0.5, minshift)
        m[i, :] = nyquist * np.random.ranf((1, Q))
        maxshift = np.max(np.max(np.sqrt(d2)))
        s[i, :] = old_div(1., np.abs(maxshift * np.random.ranf((1, Q))))
    hypinit[:Q] = w
    hypinit[Q + np.arange(0, Q * D)] = np.squeeze(m[:].T)
    hypinit[Q + Q * D + np.arange(0, Q * D)] = np.squeeze(s[:].T)

    return list(hypinit)

# Written by Srikanth Gadicherla https://github.com/imsrgadich
# Source: https://github.com/imsrgadich/gprsm/blob/master/gprsm/spectralmixture.py
def init_sm_hyper_v2(train_x, train_y, num_mixtures):
    """
    For initialization of the parameters for the Spectral Mixture
    Kernel.
    :param train_x: input data
    :param train_y: target data
    :param num_mixtures: number of mixtures
    :return: param_name       dimensions
             ----------       ----------
             mixture weights| num_mixtures x 1
             mixture means  | num_mixtures x input_dim
             mixture scales | input_dim x num_mixtures
    """
    assert isinstance(num_mixtures, int)
    assert train_x.shape[0] == train_y.shape[0]

    input_dim = np.shape(train_x)[1]  # type: int

    if np.size(train_x.shape) == 1:
        train_x = np.expand_dims(train_x ,-1)

    if np.size(train_x.shape) == 2:
        train_x = np.expand_dims(train_x ,0)

    train_x_sort = np.copy(train_x)
    train_x_sort.sort(axis=1)

    max_dist = np.squeeze(train_x_sort[: ,-1, :] - train_x_sort[: ,0, :])

    min_dist_sort = np.squeeze(np.abs(train_x_sort[: ,1:, :] - train_x_sort[: ,:-1, :]))
    min_dist = np.zeros([input_dim] ,dtype=float)

    # min of each data column could be zero. Hence, picking minimum which is not zero
    for ind in np.arange(input_dim):
        try:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort[:,ind] > 0), axis=1), ind]
        except:
            min_dist[ind] = min_dist_sort[np.amin(np.where(min_dist_sort > 0), axis=1)]


    # for random restarts during batch processing. We need to initialize at every
    # batch. Lock the seed here.
    seed= np.random.randint(low=1 ,high=2**31)
    np.random.seed(seed)

    # Inverse of lengthscales should be drawn from truncated Gaussian |N(0, max_dist^2)|
    # dim: Q x D
    # mixture_scales = tf.multiply(,tf.cast(max_dist,dtype=tf.float32)**(-1)

    mixture_scales = (np.multiply(np.abs(np.random.randn(num_mixtures,input_dim)),
                                         np.expand_dims(max_dist ,axis=0)))**(-1)

    # Draw means from Unif(0, 0.5 / minimum distance between two points), dim: Q x D
    # the nyquist is half of maximum frequency. TODO
    nyquist = np.divide(0.5,min_dist)
    mixture_means = np.multiply(np.random.rand(num_mixtures,input_dim),\
                                                       np.expand_dims(nyquist,0))
    mixture_means[0,:] = 0

    # Mixture weights should be roughly the std  of the y values divided by
    # the number of mixtures
    # dim: 1 x Q
    mixture_weights= np.divide(np.std(train_y,axis=0),num_mixtures)*np.ones(num_mixtures)

    init_hyper = np.zeros(num_mixtures*3)
    init_hyper[0:num_mixtures] = np.squeeze(np.asarray(mixture_weights))
    init_hyper[num_mixtures:num_mixtures*2] = np.squeeze(np.asarray(mixture_means))
    init_hyper[num_mixtures*2:num_mixtures*3] = np.squeeze(np.asarray(mixture_scales.T))

    return init_hyper