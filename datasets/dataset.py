from torch.utils.data import Dataset
import numpy as np

class CloudDataset(Dataset):
    def __init__(self, files, mean, std, transforms=None):
        self.files = files
        self.mean = np.array(mean)[np.newaxis, np.newaxis, :]
        self.std = np.array(std)[np.newaxis, np.newaxis, :]
        self.transforms = transforms
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        path = self.files[idx]['path']
        meta = self.files[idx]['meta']

        data = np.load(path)["data"]
        if self.transforms:
            data = self.transforms(image=data)["image"]
        
        inputs = data[:, :, :4]
        inputs = (inputs - self.mean) / self.std
        inputs = np.transpose(inputs, (2, 0, 1))

        labels = data[:, :, 4]
        labels = labels[np.newaxis, :, :]

        return {
            'inputs': inputs.astype(np.float32),
            'meta': meta.astype(np.float32),
            'labels': labels.astype(np.float32)
        }