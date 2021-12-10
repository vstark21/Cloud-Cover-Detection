import random
import numpy as np
import torch
import config
from model import CloudModel
from utils import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

paths = [
    "/content/westeurope"
]

files = []
for name in paths:
    _files = [os.path.join(name, el) for el in os.listdir(name)]
    files.extend(_files)

train_files, val_files = train_test_split(files, test_size=0.25, 
                                          random_state=config.SEED)

model = CloudModel(n_channels=config.N_CHANNELS,
                   n_classes=config.N_CLASSES).to(config.DEVICE)
model = load_model_weights(model, config.NAME + '.pt', folder=config.OUTPUT_PATH)
model.eval()

@torch.no_grad()
def random_predict(sname="pred_0"):
    name = random.choice(val_files)
    data = np.load(name)

    inputs = data[:, :, :4]
    labels = data[:, :, 4]

    inputs = np.transpose(inputs, (2, 0, 1))
    inputs = inputs[np.newaxis, :, :, :]
    inputs = torch.Tensor(inputs)
    inputs = inputs.to(config.DEVICE)

    pred = torch.sigmoid(model(inputs)).squeeze().cpu().numpy()

    fig = plt.figure(figsize=(15, 10))

    for i in range(5):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.imshow(data[:, :, i], cmap='gray')
    
    ax = fig.add_subplot(2, 3, 6)
    ax.imshow(pred, cmap='gray')
    ax.set_title("Predicton")

    plt.savefig(f"preds/{sname}.png")
    plt.show()

for i in range(5):
    random_predict(f"pred_{i}")