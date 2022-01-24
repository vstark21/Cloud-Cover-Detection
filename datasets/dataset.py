import cv2
from torch.utils.data import Dataset
import numpy as np

class CloudDataset(Dataset):
    def __init__(
        self,
        files: list,
        mean: dict,
        std: dict,
        use_bands: list,
        transforms=None
    ):
        self.files = files
        self.mean = mean
        self.std = std
        self.use_bands = use_bands
        self.transforms = transforms
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        feat_path = self.files[idx]['feat_path']
        label_path = self.files[idx]['label_path']

        _feat = np.load(feat_path)
        data = np.zeros((512, 512, len(self.use_bands) + 1), dtype=np.float32)
        for i, key in enumerate(self.use_bands):
            _band = cv2.resize(_feat[key], (512, 512))
            data[:, :, i] = (_band - self.mean[key]) / self.std[key]
        data[:, :, -1] = np.load(label_path)['label']
        if self.transforms:
            data = self.transforms(image=data)['image']
        feat = data[:, :, :-1]
        label = data[:, :, -1]
        feat = feat.transpose(2, 0, 1)
        label = np.expand_dims(label, axis=0)
        return {
            'inputs': feat.astype(np.float32),
            'labels': label.astype(np.float32)
        }