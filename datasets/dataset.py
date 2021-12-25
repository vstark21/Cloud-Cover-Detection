from torch.utils.data import Dataset
import numpy as np

class CloudDataset(Dataset):
    def __init__(self, paths, mean, std):
        self.files = paths
        self.mean = mean
        self.std = std
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])["data"]
        
        inputs = data[:, :, :4]
        inputs = (inputs - self.mean[np.newaxis, np.newaxis, :]) / self.std[np.newaxis, np.newaxis, :]
        inputs = np.transpose(inputs, (2, 0, 1))

        labels = data[:, :, 4]
        labels = labels[np.newaxis, :, :]

        return {
            'inputs': inputs.astype(np.float32),
            'labels': labels.astype(np.float32)
        }