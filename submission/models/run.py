from SwinSegmenter import SwinSegmenter
import config 
import torch

def count_parameters(model, all=False):
    """
    Counts the parameters of a model.
    Args:
        model (torch model): Model to count the parameters of.
        all (bool, optional):  Whether to count not trainable parameters. Defaults to False.
    Returns:
        int: Number of parameters.
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

device = 'cpu' if torch.cuda.is_available() else 'cpu'
img_size = config.model['backbone']['img_size']
a = torch.randn((2, 3, img_size, img_size)).to(device)

model = SwinSegmenter(config.model)
for k, v in model(a).items():
    print(k, v.shape)

print(count_parameters(model))
print(count_parameters(model, all=True))
