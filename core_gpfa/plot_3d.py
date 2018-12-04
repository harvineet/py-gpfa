# Plotting extracted trajectories
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['legend.fontsize'] = 10
plt.rc('text', usetex=True)

def plot_3d(seq, xspec='x_orth', dims_to_plot=[0,1,2]):

    n_plot_max = 20
    red_trials = [] # TODO

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    for n in range(min(len(seq), n_plot_max)):
        dat_xspec = getattr(seq[n], xspec)
        dat = dat_xspec[dims_to_plot,:]
        T   = [trial.T for trial in seq]
        ax.plot(dat[0,:], dat[1,:], dat[2,:], color='grey', marker='.', markersize=4)

    # Source: https://matplotlib.org/users/usetex.html
    ax.set_xlabel(r'$\tilde{x}_{1,:}$', fontsize=24)
    ax.set_ylabel(r'$\tilde{x}_{2,:}$', fontsize=24)
    ax.set_zlabel(r'$\tilde{x}_{3,:}$', fontsize=24)

    # Written by Remy F (https://stackoverflow.com/users/1840524/remy-f)
    # Source: https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    # Create cubic bounding box to simulate equal aspect ratio
    X = dat[0,:]
    Y = dat[1,:]
    Z = dat[2,:]
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
    for xb, yb, zb in zip(Xb, Yb, Zb):
       ax.plot([xb], [yb], [zb], 'w')

    plt.show()