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
        chip_id = self.files[idx]['chip_id']
        feat_path = self.files[idx]['feat_path']
        label_path = self.files[idx]['label_path']

        _feat = np.load(feat_path)
        feat = []
        for key in self.use_bands:
            try:
                _band = cv2.resize(_feat[key], (512, 512))
            except Exception as e:
                print(f"Following error raised due to {chip_id}: {e}")

            _band = (_band - self.mean[key]) / self.std[key]
            feat.append(_band)
        feat = np.stack(feat, axis=0)
        label = np.load(label_path)['label']
        if self.transforms:
            data = self.transforms(feat=feat, label=label)
            feat = data['feat']
            label = data['label']
        feat = np.expand_dims(feat, axis=0)
        label = np.expand_dims(label, axis=0)
        return {
            'inputs': feat.astype(np.float32),
            'labels': label.astype(np.float32)
        }