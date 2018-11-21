# Simulate data from a known GPFA process

from Seq_Data_Class import Model_Specs, Trial_Class, Param_Class
import numpy as np

# Parameters
# TODO

# Simulate
def sample_data():
    # TODO
    seq = [Trial_Class(i, 20, 1, np.random.rand(53,20)) for i in range(10)]

    return seq

# Save to file
def save_data(filepath):
    # TODO
    print("Saved file at", filepath)

# Load from file
def load_data(filepath):
    # TODO
    model_data = Model_Specs()
    model_data.data_from_mat(filepath)
    data = model_data.data

    return data

# Load parameters from mat file
def load_params(filepath):
    params = Param_Class()
    params.params_from_mat(filepath)

    return params

if __name__ == "__main__":
    print("Simulating data")
    print(sample_data())