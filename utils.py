import torch
import math
import numpy as np

from typing import List, Union, Tuple

def generate_style_mixes(
        mapping_network, 
        target_resolution : int, 
        batch_size : int, 
        device : str, 
        style_mixing_prob : float = 0.9,
        truncation_psi : float = 1.0,
        update_w_ema : bool = False) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor]], List[int]]:
    
    w_sm, ws, crossovers = [], [], []
    i = 0

    l = int(2 * (math.log2(target_resolution) - 1))

    while i < batch_size:
        x = torch.rand(()).item()
        if x < style_mixing_prob:
            z = torch.randn((2, mapping_network.latent_dim)).to(device)
            w = mapping_network(z, truncation_psi = truncation_psi, update_w_ema = update_w_ema)

            """
            w1 is to be used in range [0, crossover_point], w2 is to be used in range [crossover_point + 1, l - 1].
            Taking randint(0, generator_layers) ensures that at least one part of both vectors will be used in generated style mixing.
            """
            crossover_point = torch.randint(0, l - 1, (1,)).item() 
            crossovers.append(crossover_point)
            ws.append((w[0], w[1]))
            w = torch.cat((w[0:1].expand(crossover_point + 1, -1), w[1:2].expand(l - crossover_point - 1, -1)), dim = 0)
            w_sm.append(w)

        else:
            z = torch.randn((1, mapping_network.latent_dim)).to(device)
            w = mapping_network(z, truncation_psi = truncation_psi, update_w_ema = update_w_ema).expand(l, -1)
            crossovers.append(0)
            ws.append((w[0],))
            w_sm.append(w)

        i += 1
    
    return torch.stack(w_sm, dim = 1), ws, crossovers

def new_style_mixing(mapping_network, num_samples, device, style_mixing_prob, num_ws : int, update_w_ema : bool = True, truncation_psi : float = 1):
    z = torch.randn((num_samples, mapping_network.latent_dim)).to(device)
    w = mapping_network(z, update_w_ema = update_w_ema, truncation_psi = truncation_psi).unsqueeze(0).repeat(num_ws, 1, 1)

    if style_mixing_prob > 0:
        cutoff = torch.empty([], dtype = torch.int64, device = device).random_(1, num_ws) # Random integer in range [1, num_ws - 1]
        cutoff = torch.where(torch.rand([], device = device) < style_mixing_prob, cutoff, torch.full_like(cutoff, num_ws)) # Uses same cutoff for all samples in the batch
        # Do not update w_ema in tyle mixing pass, as per official implementation:
        # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/d72cc7d041b42ec8e806021a205ed9349f87c6a4/training/loss.py#L45
        w[cutoff:] = mapping_network(torch.randn_like(z), update_w_ema = False, truncation_psi = truncation_psi).unsqueeze(0).repeat(num_ws - cutoff, 1, 1)

    # TODO: Return tuple, obtained style mixes, style vectors that generated the mixes, and cutoff point
    return w, None, None

def generate_noise(target_resolution : int, batch_size : int, device : str) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    z = [torch.randn(batch_size, 1, 4, 4).to(device)] + \
        [(torch.randn(batch_size, 1, 2 ** i, 2 ** i).to(device), torch.randn(batch_size, 1, 2 ** i, 2 ** i).to(device)) for i in range(3, int(math.log2(target_resolution) + 1))]
    
    return z

def samples_to_grid(samples : torch.Tensor, num_rows : int) -> np.ndarray:
    samples = ((samples + 1) * 127.5).cpu().detach().numpy()
    samples = np.rint(samples).clip(0, 255).astype(np.uint8)
    num_samples = samples.shape[0]

    assert num_samples % num_rows == 0

    gh = num_samples // num_rows
    gw = num_samples // gh
    _, C, H, W = samples.shape

    samples = samples.reshape(gh, gw, C, H, W)
    samples = samples.transpose(0, 3, 1, 4, 2) # (gh, H, gw, W, C)
    samples = samples.reshape(gh * H, gw * W, C)
    return samples

def generate_samples(
        generator, 
        mapping_network, 
        target_resolution : int, 
        device : str, 
        num_samples : int,
        num_ws : int,
        style_mixing_prob : float = 0.9,
        truncation_psi : float = 1.0,
        update_w_ema : bool = False,
        num_generated_rows : int = 1):
    
    w, _, _ = new_style_mixing(mapping_network, 
                                num_samples, 
                                device, 
                                style_mixing_prob,
                                num_ws,
                                update_w_ema = update_w_ema,
                                truncation_psi = truncation_psi)
        
    fake_samples = generator(w, generate_noise(target_resolution, num_samples, device))
    return samples_to_grid(fake_samples, num_generated_rows)