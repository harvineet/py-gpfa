# Simulate data from a known GPFA process

from Seq_Data_Class import Seq_Data_Class
import numpy as np

# Parameters
# TODO

# Simulate
# TODO
# sample output
seq = [Seq_Data_Class(i, 20, 1, np.random.rand(53,20)) for i in range(10)]
print(seq)

# Save to file
# TODO

# Load from file
def load_data():
    # TODO
    return None