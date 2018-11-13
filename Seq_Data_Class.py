# Class to store data GPFA model
# Usage:
#
# seq = Seq_Data_Class(trial_id, T, seg_id, y)
# seq.y = np.zeros((10,1))

class Seq_Data_Class():
    def __init__(self, trial_id, T, seg_id, y):
        self.trial_id = trial_id
        self.T = T
        self.seg_id = seg_id
        self.y = y
    # Function to print objects
    def __repr__(self):
        return "("+",".join(map(str, [self.trial_id, self.T, self.seg_id] + list(self.y.shape)))+")"