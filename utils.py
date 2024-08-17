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

def generate_noise(target_resolution : int, batch_size : int, device : str) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    z = [torch.randn(batch_size, 1, 4, 4).to(device)] + \
        [(torch.randn(batch_size, 1, 2 ** i, 2 ** i).to(device), torch.randn(batch_size, 1, 2 ** i, 2 ** i).to(device)) for i in range(3, int(math.log2(target_resolution) + 1))]
    
    return z

def generate_samples(
        generator, 
        mapping_network, target_resolution : int, 
        num_samples : int, 
        device : str, 
        style_mixing_prob : float = 0.9,
        truncation_psi : float = 1.0,
        update_w_ema : bool = False,
        num_generated_rows : int = 1):
    
    w, _, _ = generate_style_mixes(mapping_network, 
                                   target_resolution, 
                                   num_samples, 
                                   device, 
                                   style_mixing_prob = style_mixing_prob, 
                                   truncation_psi = truncation_psi, 
                                   update_w_ema = update_w_ema)
    
    fake_samples = generator(w, generate_noise(target_resolution, num_samples, device))
    fake_samples = ((fake_samples + 1) * 127.5).cpu().detach().numpy()
    fake_samples = np.rint(fake_samples).clip(0, 255).astype(np.uint8)

    assert num_samples % num_generated_rows == 0

    gh = num_samples // num_generated_rows
    gw = num_samples // gh
    _, C, H, W = fake_samples.shape

    fake_samples = fake_samples.reshape(gh, gw, C, H, W)
    fake_samples = fake_samples.transpose(0, 3, 1, 4, 2) # (gh, H, gw, W, C)
    fake_samples = fake_samples.reshape(gh * H, gw * W, C)
    return fake_samples