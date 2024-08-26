import argparse
import torch
import torchvision
import tqdm
import numpy as np

from model import MappingNetwork, Generator
from dataset import Dataset, get_inception_data_loader
from typing import Union, Tuple
from scipy import linalg

# Some inspiration taken from: https://github.com/mseitzer/pytorch-fid

class InpcetionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = device
        self.inception_v3 = torchvision.models.inception_v3(weights = torchvision.models.Inception_V3_Weights.DEFAULT).eval()
        # Replace final fully connected layer with identity mapping to obtain feature maps from last convolutional layer
        self.inception_v3.fc = torch.nn.Identity()

    @torch.no_grad
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        return self.inception_v3(X)

def parse_args():
    args = argparse.ArgumentParser()
    return args.parse_args()

'''
- Compute FID activations on a given path, create dataset and data loader from that parciular path. Stack activations, transform into numpy vector and optionally 
  cache that somwhere to avoid re-computation.

- 
'''

def compute_statistics_path(path : str, model : InpcetionV3, device : str, batch_size : int = 32, cache_path : Union[str, None] = None) -> Tuple[np.ndarray, np.ndarray]:
    dataset = Dataset(path, is_inception = True, alpha = 0.01)
    dl = get_inception_data_loader(dataset, batch_size = batch_size)
    
    acts = torch.tensor([], device = device, dtype = torch.float32)

    for X in tqdm.tqdm(dl, desc = f"Comptuing FID statistics over images in {path}."):
        X = X.to(device)
        assert X.shape[1] == 3, "FID does not support input channels other than 3"
        acts = torch.cat([acts, model(X)], dim = 0)
    
    # Conversion of tensor to numpy requires that tensor is on CPU
    acts = acts.reshape((-1, 2048)).cpu().numpy()
    print("Computing inception statistics.")
    mu = acts.mean(axis = 0)
    var = np.cov(acts, rowvar = False)
    
    print("Statistics computation done!")
    if cache_path is not None:
        print(f"Saving statistics to {cache_path}.")
        np.savez_compressed(cache_path, mu = mu, var = var)

    return mu, var

def compute_fid(mu1 : np.ndarray, var1 : np.ndarray, mu2 : np.ndarray, var2 : np.ndarray):
    pass
    t1 = ((mu1 - mu2) ** 2).sum()

    # Covariance matrix is PSD and symmetric
    # Symmetric matrices are always diagonalizable
    # If A and B are PSD and simultaneously diagonalizable, then they commute, and since they commute, product AB is PSD as well
    var_sqrt = linalg.sqrtm(var1.dot(var2), disp = False)
    if not np.isfinite(var_sqrt).all():
        pass

if __name__ == "__main__":
    device = "cuda"
    model = InpcetionV3().to(device)
    compute_statistics_path("celeba_128_eval", model, device, cache_path = "test.npz")

    with np.load("test.npz") as f:
        mu, sigma = f["mu"], f["var"]