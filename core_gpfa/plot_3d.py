# Plotting extracted trajectories
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from data_simulator import load_params
from core_gpfa.postprocess import orthogonalize

mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.figsize'] = [10, 6]
# mpl.rcParams['figure.dpi'] = 300
plt.rc('text', usetex=True)

def get_colors(n_colors):
    return plt.cm.twilight(np.linspace(0,1-1/n_colors,n_colors))

# TODO sample multiple seq_id and plot
def plot_3d(seq, xspec='x_orth', dims_to_plot=[0,1,2], output_file='output/plot_3d.pdf'):

    n_plot_max = 3
    n_plot_max_per_seqid = 2
    red_trials = [] # TODO

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.set_aspect('equal')

    # Get number of unique seq_id for coloring
    list_seq_id = []
    for n in range(len(seq)):
        if hasattr(seq[0], 'seq_id'):
            list_seq_id.append(seq[n].seq_id)
    uniq_seq_id = set(list_seq_id)

    # Select trials with different seq_id
    trial_ids_plot = []
    for sid in uniq_seq_id:
        ids = [i for i,s in enumerate(list_seq_id) if s==sid]
        trial_ids_plot+=ids[:n_plot_max_per_seqid]

    # TODO Create colormap for len(uniq_seq_id) colors
    # Category10 color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    cond_label = {0:'BZD', 1:'EtGI', 2:'Air', 3:'Hex', 4:'MVT0'}

    # for n in range(min(len(seq), n_plot_max)):
    for n in trial_ids_plot:
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
            # ax.plot(x_1, x_2, x_3, color=colors[min(seq[n].seq_id, len(colors)-1)], marker='.',\
            #          markersize=4, label='Cond: '+str(seq[n].seq_id)+', Trial: '+str(seq[n].trial_id))

            # ax.plot(x_1, x_2, x_3, color=colors[min(seq[n].seq_id, len(colors)-1)], alpha=0.6, marker='.',\
            #          markersize=4, label='Cond: '+str(cond_label[min(seq[n].seq_id, len(cond_label)-1)])+', Trial: '+str(seq[n].trial_id))
            if seq[n].seq_id==2:
                ax.plot(x_1, x_2, x_3, color=colors[min(seq[n].seq_id, len(colors)-1)], alpha=0.6, marker='.',\
                         markersize=4, label='Cond: '+str(cond_label[min(seq[n].seq_id, len(cond_label)-1)])+', Trial: '+str(seq[n].trial_id))
            else:
                ax.plot(x_1, x_2, x_3, color='grey', alpha=0.6, marker='.',\
                         markersize=4, label='Cond: Not Air, Trial: '+str(seq[n].trial_id))

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

    # Remove plane color and grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)

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
def plot_1d_error(seq, xspec='x_orth', bin_width=50, output_file='output/plot_1d_error.pdf', cov_type='rbf'):
    # Plot prediction for each latent dimension over time

    n_plot_max = 5
    n_plot = min(len(seq), n_plot_max)
    colors = get_colors(n_plot)
    red_trials = [] # TODO

    n_cols = 4

    fig = plt.figure()
    ax = fig.gca()

    X_all = np.concatenate([getattr(trial, xspec) for trial in seq], 1)
    x_max = np.ceil(10 * np.max(np.abs(X_all))) / 10

    # perform SVD on true_X_all to get orthogonalized "acutal" latent space
    params = load_params('input/example_params_{}.mat'.format(cov_type))
    true_X_all = np.concatenate([trial.x for trial in seq], 1)
    (true_X_orth, true_C_orth, _) = orthogonalize(true_X_all, params.C, full_mat=True)

    T_max = np.max([trial.T for trial in seq])
    xtk_step = np.ceil(T_max/25.0) * 5
    xtk = np.arange(0, T_max, xtk_step)
    xtkl = np.arange(0, (T_max-1)*bin_width+1, xtk_step*bin_width)
    ytk = [-x_max, 0, x_max]

    n_rows = int(np.ceil(X_all.shape[0]*1.0 / n_cols))

    for k in range(X_all.shape[0]):  # for each latent dimension
        ax = plt.subplot(n_rows, n_cols, k+1)
        startT = 0
        endT = 0

        acts = [[]] * n_plot
        sq_errs = [[]] * n_plot
        flipped_sq_errs = [[]] * n_plot

        for n in range(n_plot):  # for each trial (up to n_plot_max trials)

            dat = getattr(seq[n], xspec)
            T = seq[n].T

            startT = endT
            endT = startT + T
            
            pred_mean = np.squeeze(np.asarray(dat[k,:])) # dat is originally in matrix format. Can also use np.ravel(dat[k,:])

            var = seq[n].Vsm[k,k,:] # CHECK if correct
            error_bar = np.sqrt(var) # CHECK if correct
            # error_bar = 2 * np.sqrt(np.diag(var)) # CHECK if correct
            # Plot error
            ax.fill_between(range(T), pred_mean + error_bar, pred_mean - error_bar, alpha=0.1, color=colors[n])

            # Plot mean
            ax.plot(range(T), pred_mean, linewidth=1, color=colors[n], label='Predicted mean')

            # Plot actual
            # Dimension of predicted latent states not equal to true latent dimensions 
            if seq[n].x.shape[0] != X_all.shape[0]:
                print("True and predicted latent state dimensions do not match")
                break

            acts[n] = np.squeeze(true_X_orth[k,startT:endT])
            # sign of the SVD is arbitrary for each latent dimension, so check which gives us a closer match to the data
            # however, must be the same for each trial within a given latent dimension, so compute sq errors and plot
            # whichever one minimizes the sq error across all trials
            sq_errs[n] = np.sum(np.power(acts[n] - pred_mean, 2))
            flipped_sq_errs[n] = np.sum(np.power(-1.*acts[n] - pred_mean, 2))

        dim_sign = 1.
        if np.sum(flipped_sq_errs) < np.sum(sq_errs):
            dim_sign = -1.
        [ax.plot(range(T), dim_sign*acts[n], marker='x', linewidth=1, color=colors[n]) for n in range(n_plot)]

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
