# Plotting extracted trajectories
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['legend.fontsize'] = 10

def plot_3d(seq, xspec='x_orth', dims_to_plot=[0,1,2]):

    n_plot_max = 20
    red_trials = [] # TODO

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    for n in range(min(len(seq), n_plot_max)):
        dat_xspec = getattr(seq[n], xspec)
        dat = dat_xspec[dims_to_plot,:]
        T   = [trial.T for trial in seq]
        ax.plot(dat[0,:], dat[1,:], dat[2,:])

    # TODO equal axes length, labels

    plt.show()