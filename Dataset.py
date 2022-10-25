from torch.utils.data import Dataset

class MelDataLoader(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label