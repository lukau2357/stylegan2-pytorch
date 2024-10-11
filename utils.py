import torch
import math
import numpy as np
import os
import json

from typing import List, Union, Tuple
from torch.nn.parallel import DistributedDataParallel
from dataset import Dataset

def find_label(root_path, metric, compare_operator, assume_first = True):
    res_label, res_metric = None, None
    
    for label in os.listdir(root_path):
        try:
            with open(os.path.join(root_path, label, ".metadata.json"), "r", encoding = "utf-8") as f:
                local_data = json.load(f)

            if assume_first and res_metric is None:
                res_label = label
                res_metric = local_data[metric]
                continue

            if metric in local_data and compare_operator(local_data[metric], res_metric):
                res_label = label
                res_metric = local_data[metric]

        except Exception as e:
            continue
    
    # Warning log: res_label is None
    return res_label

def style_mixing(mapping_network, num_ws : int, num_samples : int, device : str, style_mixing_prob : float, 
                 update_w_ema : bool = True, truncation_psi : float = 1, w_mean : Union[None, torch.Tensor] = None):
    
    latent_dim = mapping_network.module.latent_dim if isinstance(mapping_network, DistributedDataParallel) else mapping_network.latent_dim
    z = torch.randn((num_samples, latent_dim), device = device)
    w = mapping_network(z, update_w_ema = update_w_ema, truncation_psi = truncation_psi, w_mean_estimate = w_mean).unsqueeze(0).repeat(num_ws, 1, 1)

    if style_mixing_prob > 0:
        cutoff = torch.empty([], dtype = torch.int64, device = device).random_(1, num_ws) # Random integer in range [1, num_ws - 1]
        cutoff = torch.where(torch.rand([], device = device) < style_mixing_prob, cutoff, torch.full_like(cutoff, num_ws)) # Uses same cutoff for all samples in the batch
        # Do not update w_ema in tyle mixing pass, as per official implementation:
        # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/loss.py#L45
        w[cutoff:] = mapping_network(torch.randn_like(z), update_w_ema = False, truncation_psi = truncation_psi, w_mean_estimate = w_mean).unsqueeze(0).repeat(num_ws - cutoff, 1, 1)

    # TODO: Return tuple, obtained style mixes, style vectors that generated the mixes, and cutoff point?
    return w, None, None

def generate_noise(target_resolution : int, batch_size : int, device : str) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    z = [torch.randn((batch_size, 1, 4, 4), device = device)] + \
        [(torch.randn((batch_size, 1, 2 ** i, 2 ** i), device = device), torch.randn((batch_size, 1, 2 ** i, 2 ** i), device = device)) for i in range(3, int(math.log2(target_resolution) + 1))]
    
    return z

def samples_to_grid(samples : torch.Tensor, num_rows : int) -> np.ndarray:
    samples = ((samples + 1) * 127.5).cpu().detach().numpy() # Back to (0, 255) with clipping
    samples = np.rint(samples).clip(0, 255).astype(np.uint8)
    B, C, H, W = samples.shape

    assert B % num_rows == 0

    gh = B // num_rows
    gw = B // gh
    
    samples = samples.reshape(gh, gw, C, H, W)
    samples = samples.transpose(0, 3, 1, 4, 2) # (gh, H, gw, W, C)
    samples = samples.reshape(gh * H, gw * W, C)
    return samples

def generate_samples(
        generator, 
        mapping_network, 
        device : str, 
        num_samples : int,
        style_mixing_prob : float = 0.9,
        truncation_psi : float = 1.0,
        update_w_ema : bool = False,
        num_generated_rows : int = 1,
        w_estimate_samples : int = 20000,
        compute_truncation_base : bool = False):
    
    target_resolution = generator.module.image_size if isinstance(generator, DistributedDataParallel) else generator.image_size
    estimate_w = mapping_network.module.estimate_w if isinstance(mapping_network, DistributedDataParallel) else mapping_network.estimate_w
    w_mean = None

    latent_dim = mapping_network.module.latent_dim if isinstance(mapping_network, DistributedDataParallel) else mapping_network.latent_dim
    num_ws = 2 * generator.module.num_layers if isinstance(generator, DistributedDataParallel) else 2 * generator.num_layers
    truncation_base = None

    if not estimate_w:
        z = torch.randn((w_estimate_samples, latent_dim), device = device)
        w_mean = mapping_network(z).mean(dim = 0)
        del z
    
    w, _, _ = style_mixing(mapping_network, num_ws, num_samples, device, style_mixing_prob,
                            update_w_ema = update_w_ema,
                            truncation_psi = truncation_psi,
                            w_mean = w_mean)  
      
    fake_samples = generator(w, generate_noise(target_resolution, num_samples, device))
    grid = samples_to_grid(fake_samples, num_generated_rows)

    if compute_truncation_base:
        if estimate_w:
            w_mean = mapping_network.module.w_ema if isinstance(mapping_network, DistributedDataParallel) else mapping_network.w_ema
        
        w_mean = w_mean.unsqueeze(0).repeat(2 * generator.num_layers, 1, 1)
        truncation_base = generator(w_mean, generate_noise(target_resolution, 1, device))[0]
        truncation_base = ((truncation_base + 1) * 127.5).cpu().detach().numpy() # Back to (0, 255) with clipping
        truncation_base = np.rint(truncation_base).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        return grid, truncation_base
    
    return grid