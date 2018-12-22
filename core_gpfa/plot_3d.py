# Plotting extracted trajectories
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = [10, 6]
# mpl.rcParams['figure.dpi'] = 300
plt.rc('text', usetex=True)

# TODO sample multiple seq_id and plot
def plot_3d(seq, xspec='x_orth', dims_to_plot=[0,1,2], output_file='output/plot_3d.pdf'):

    n_plot_max = 20
    red_trials = [] # TODO

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    # Get number of unique seq_id for coloring
    uniq_seq_id = []
    for n in range(len(seq)):
        if hasattr(seq[0], 'seq_id'):
            uniq_seq_id.append(seq[n].seq_id)
    uniq_seq_id = set(uniq_seq_id)
    # TODO Create colormap for len(uniq_seq_id) colors
    # Category10 color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

    for n in range(min(len(seq), n_plot_max)):
        dat_xspec = getattr(seq[n], xspec)
        dat = dat_xspec[dims_to_plot,:]
        T = seq[n].T
        x_1 = np.squeeze(np.asarray(dat[0,:]))
        x_2 = np.squeeze(np.asarray(dat[1,:]))
        x_3 = np.squeeze(np.asarray(dat[2,:]))

        if len(uniq_seq_id)==0:
            ax.plot(x_1, x_2, x_3, color='grey', marker='.', markersize=4)
        else:
            # Color based on seq_id
            ax.plot(x_1, x_2, x_3, color=colors[min(seq[n].seq_id, len(colors)-1)], marker='.',\
                     markersize=4, label='Cond: '+str(seq[n].seq_id)+', Trial: '+str(seq[n].trial_id))

    if len(uniq_seq_id)>0:
        # Remove duplicate labels
        # Source: https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
        # handles, labels = plt.gca().get_legend_handles_labels()
        # print(handles, labels)
        # uniq_by_label = {}
        # for handle, label in zip(handles, labels):
        #     uniq_by_label[label] = handle
        # print(uniq_by_label)
        # ax.legend(uniq_by_label.values(), uniq_by_label.keys())
        
        # Shrink width by 20% and plot legend on right
        # Source: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # ax.legend()

    # Source: https://matplotlib.org/users/usetex.html
    if xspec=='x_orth':
        ax.set_xlabel(r'$\tilde{x}_{1,:}$', fontsize=18)
        ax.set_ylabel(r'$\tilde{x}_{2,:}$', fontsize=18)
        ax.set_zlabel(r'$\tilde{x}_{3,:}$', fontsize=18)
    elif xspec=='xsm':
        ax.set_xlabel(r'$x_{1,:}$', fontsize=18)
        ax.set_ylabel(r'$x_{2,:}$', fontsize=18)
        ax.set_zlabel(r'$x_{3,:}$', fontsize=18)

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

    plt.tight_layout()
    plt.savefig(output_file+'_plot_3d.pdf', transparent=True)
    plt.show(block=False)

def plot_1d(seq, xspec='x_orth', bin_width=20, output_file='output/plot_1d.pdf'):
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

            pred_mean = np.squeeze(np.asarray(dat[k,:]))
            ax.plot(range(T), pred_mean, linewidth=1, color='grey')

        ax.set_xlim([0, T_max])
        ax.set_ylim([1.1*min(ytk), 1.1*max(ytk)])

        # Source: https://matplotlib.org/users/usetex.html
        if xspec=='x_orth':
            ax.set_title(r'$\tilde{\mathbf{x}}_{%d,:}$' % (k+1), fontsize=18)
        elif xspec=='xsm':
            ax.set_title(r'$\mathbf{x}_{%d,:}$' % (k+1), fontsize=18)

        ax.set_xticks(xtk)
        ax.set_xticklabels(xtkl)

        ax.set_yticks(ytk)
        ax.set_yticklabels(ytk)
        
        ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(output_file+'_plot_1d.pdf', transparent=True)
    plt.show(block=False)

# Uncertainty plots based on code by Martin Krasser https://github.com/krasserm
# Source https://github.com/krasserm/bayesian-machine-learning/blob/master/gaussian_processes_util.py
def plot_1d_error(seq, xspec='x_orth', bin_width=20, output_file='output/plot_1d_error.pdf'):
    # Plot prediction for each latent dimension over time

    n_plot_max = 5
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
            
            pred_mean = np.squeeze(np.asarray(dat[k,:])) # dat is originally in matrix format. Can also use np.ravel(dat[k,:])

            var = seq[n].Vsm[k,k,:] # CHECK if correct
            error_bar = 2 * np.sqrt(var) # CHECK if correct
            # error_bar = 2 * np.sqrt(np.diag(var)) # CHECK if correct
            # Plot error
            ax.fill_between(range(T), pred_mean + error_bar, pred_mean - error_bar, alpha=0.1)

            # Plot mean
            ax.plot(range(T), pred_mean, linewidth=1, color='grey', label='Predicted mean')

            # Plot actual
            # Dimension of predicted latent states not equal to true latent dimensions 
            if seq[n].x.shape[0] != X_all.shape[0]:
                print("True and predicted latent state dimensions do not match")
                break
            
            act = np.squeeze(np.asarray(seq[n].x[k,:]))
            ax.plot(range(T), act, marker='x', linewidth=1, color='red')

        ax.set_xlim([0, T_max])
        ax.set_ylim([1.1*min(ytk), 1.1*max(ytk)])

        # Source: https://matplotlib.org/users/usetex.html
        if xspec=='x_orth':
            ax.set_title(r'$\tilde{\mathbf{x}}_{%d,:}$' % (k+1), fontsize=18)
        elif xspec=='xsm':
            ax.set_title(r'$\mathbf{x}_{%d,:}$' % (k+1), fontsize=18)

        ax.set_xticks(xtk)
        ax.set_xticklabels(xtkl)

        ax.set_yticks(ytk)
        ax.set_yticklabels(ytk)
        
        ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.savefig(output_file+'_plot_1d_error.pdf', transparent=True)
    plt.show(block=False)
