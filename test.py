from tests.test_inference import test_inference, test_orthonormalize
from tests.test_em import test_em
from data_simulator import load_data
from Seq_Data_Class import Param_Class
from core_gpfa.plot_3d import plot_3d, plot_1d, plot_1d_error

if __name__ == "__main__":

    # Load data, params from sample file
    INPUT_FILE = '../fake_data_w_genparams.mat' # '../em_input_new.mat'
    seq = load_data(INPUT_FILE)
    params = Param_Class()
    params.params_from_mat(INPUT_FILE)
    params.learnKernelParams = True
    params.learnGPNoise = False
    params.RforceDiagonal = True

    # Test for em
    # res = test_em(params, seq, kernSDList = 30)

    # Test for inference
    seq, LL = test_inference(seq, params)
    print("LL", LL)
    print("xsm", seq[0].xsm)

    # Test for orthonormalization
    est_params, seq, _ = test_orthonormalize(LL, params, seq)

    print("x_orth", seq[0].x_orth)
    print("C_orth", est_params.C_orth)

    # Test for 3d plot
    plot_3d(seq, 'x_orth', dims_to_plot=[0,1,2])

    # Test for 1d plot
    plot_1d(seq, 'x_orth', bin_width=20)

    # Test error plot
    plot_1d_error(seq, 'x_orth', bin_width=20)