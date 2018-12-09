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
        T = seq[n].T
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

def plot_1d(seq, xspec='x_orth', bin_width=20):
    # Plot each latent dimension over time

    n_plot_max = 20
    red_trials = [] # TODO

    n_cols = 4

    fig = plt.figure()
    ax = fig.gca()

    X_all = np.concatenate([getattr(trial, xspec) for trial in seq], 1)
    x_max = np.ceil(10 * np.max(np.abs(X_all))) / 10

    T_max = np.max([trial.T for trial in seq])
    xtk_step = np.ceil(T_max/25.0) * 5
    xtk = np.arange(0, T_max, xtk_step)
    xtkl = np.arange(0, (T_max-1)*bin_width+1, xtk_step*bin_width)
    ytk = [-x_max, 0, x_max]

    n_rows = int(np.ceil(X_all.shape[0]*1.0 / n_cols))

    for k in range(X_all.shape[0]):
        ax = plt.subplot(n_rows, n_cols, k+1)

        for n in range(min(len(seq), n_plot_max)):
            dat = getattr(seq[n], xspec)
            T = seq[n].T

            ax.plot(range(T), dat[k,:], linewidth=1, color='grey')

        ax.set_xlim([0, T_max])
        ax.set_ylim([1.1*min(ytk), 1.1*max(ytk)])

        # Source: https://matplotlib.org/users/usetex.html
        ax.set_title(r'$\tilde{\mathbf{x}}_{%d,:}$' % (k+1), fontsize=14)

        ax.set_xticks(xtk)
        ax.set_xticklabels(xtkl)

        ax.set_yticks(ytk)
        ax.set_yticklabels(ytk)
        
        ax.set_xlabel('Time')
    
    plt.show()

        