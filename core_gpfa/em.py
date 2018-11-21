# EM implementation

from data_simulator import load_params

# Run EM
def em(current_params, seq):
    # TODO
    params = load_params('../em_input.mat')

    # est_params, seq_train_cut, LLcut, iter_time
    return params, seq, None, None