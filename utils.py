import torch
import math
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