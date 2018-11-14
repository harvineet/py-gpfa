# Simulate data from a known GPFA process

from Seq_Data_Class import Seq_Data_Class
import numpy as np

# Parameters
# TODO

# Simulate
# TODO
# sample output
def sample_data():
	seq = [Seq_Data_Class(i, 20, 1, np.random.rand(53,20)) for i in range(10)]
	return seq

# Save to file
# TODO

# Load from file
def load_data():
    # TODO
    return None

if __name__ == "__main__":
    print("Running")
    print(sample_data())