from torch.utils.data import Dataset
import numpy as np

class CloudDataset(Dataset):
    def __init__(
        self, 
        paths, 
        data_mean, 
        data_std,
        model_mean,
        model_std
    ):
        self.files = paths
        self.data_mean = np.array(data_mean)[np.newaxis, np.newaxis, :]
        self.data_std = np.array(data_std)[np.newaxis, np.newaxis, :]
        self.model_mean = np.array(model_mean)[np.newaxis, np.newaxis, :]
        self.model_std = np.array(model_std)[np.newaxis, np.newaxis, :]

            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])["data"]
        
        inputs = data[:, :, :4]
        inputs = (inputs - self.data_mean) / self.data_std
        inputs = (inputs * self.model_std) + self.model_mean
        inputs = np.transpose(inputs, (2, 0, 1))

        labels = data[:, :, 4]
        labels = labels[np.newaxis, :, :]

        return {
            'inputs': inputs.astype(np.float32),
            'labels': labels.astype(np.float32)
        }