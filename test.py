from tests.test_inference import test_inference
from data_simulator import load_data
from Seq_Data_Class import Param_Class

if __name__ == "__main__":

    # Load data, params from sample file
    INPUT_FILE = '../em_input.mat'
    seq = load_data(INPUT_FILE)
    params = Param_Class()
    params.params_from_mat(INPUT_FILE)

    res = test_inference(seq, params)

    print(res)