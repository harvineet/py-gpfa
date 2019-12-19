from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error
import matplotlib.pyplot as plt
from Seq_Data_Class import Param_Class
import scipy.io as sio
from core_gpfa.postprocess import postprocess

def load_results(fname):
    # Loads a dict with keys: 'LLtrain', 'LLtest', 'params', 'seq_train', 'seq_test',
    # 'bin_width' and 'leave_one_out'
    result = sio.loadmat(fname, struct_as_record=False, squeeze_me=True)
    return result

if __name__ == "__main__":

    RUN_ID = 15
    INPUT_DIR = './output/'+str(RUN_ID)+'/'

    x_dim = 8 # latent dimension
    method = 'gpfa'
    param_cov_type = 'sm' # type of kernel: 'rbf', 'sm'
    param_Q = 3 # number of mixtures for SM
    num_folds = 3 # change to n>=2 for n-fold cross-validation
    kern_SD = 30

    # Load results from mat file
    input_file = INPUT_DIR+"/"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type
    result = load_results(input_file)

    # Orthonormalize trajectories
    # Returns results for the last run cross-validation fold, if enabled
    (est_params, seq_train, seq_test) = postprocess(result['params'], result['seq_train'],\
                                                     result['seq_test'], method, kern_SD)

    print("LL for training: %.4f, for testing: %.4f, method: %s, x_dim:%d, param_cov_type:%s, param_Q:%d"\
             % (result['LLtrain'], result['LLtest'], method, x_dim, param_cov_type, param_Q))

    # Output filenames for plots
    OUTPUT_DIR = './output/'+str(RUN_ID)+'/'
    output_file = OUTPUT_DIR+"/"+"new_"+method+"_xdim_"+str(x_dim)+"_cov_"+param_cov_type

    # Plot trajectories in 3D space
    if x_dim >=3:
        plot_3d(seq_train, 'x_orth', dims_to_plot=[0,1,2], output_file=output_file)

    # Plot each dimension of trajectory
    # plot_1d(seq_train, 'x_sm', result['bin_width'], output_file=output_file)
    plot_1d(seq_train, 'x_orth', result['bin_width'], output_file=output_file)

    # Prediction error and extrapolation plots on test set
    if len(seq_test)>0:
        # Change to 'x_orth' to get prediction error for orthogonalized trajectories
        mean_error_trials = mean_squared_error(seq_test, 'x_orth')
        print("Mean sequared error across trials: %.4f" % mean_error_trials)

        r2_trials = goodness_of_fit_rsquared(seq_test, x_dim, 'xsm')
        print("R^2 averaged across trials: %s" % np.array_str(r2_trials, precision=4))

        # # Plot each dimension of trajectory, test data
        # plot_1d(seq_test, 'x_orth', result['bin_width'])
        # Change to 'x_orth' to plot orthogonalized trajectories
        plot_1d_error(seq_test, 'xsm', result['bin_width'], output_file=output_file)

    plt.show()