import torch
import os
import copy
import losses
import time
import math

from model import Generator, Discriminator, MappingNetwork
from dataset import Dataset, get_data_loader
from typing import Union
from utils import generate_style_mixes, generate_noise

def requires_grad(model : torch.nn.Module, flag : bool):
    for param in model.parameters():
        param.requires_grad_(flag)

# For now, supports only single GPU training
# Also, no gradient accumulation, for now.
class Trainer:
    def __init__(self,
        MN : Union[MappingNetwork, None],
        G : Union[Generator, None],
        D: Union[Discriminator, None],
        root_path : str,
        train_loader : torch.utils.data.DataLoader,
        val_loader : torch.utils.data.DataLoader,
        total_steps : int,              # Number of training steps (for generator?),
        device : str,
        checkpoint_path : Union[str, None] = None,    # Checkpoint directory for the model inside root_path. If present, will load and overwrite given model if it is not None,
        load_latest : Union[bool, None] = None,
        load_best_fid_val : Union[bool, None] = None,
        loss_type : str = "vanilla",    # Use vanilla GAN loss or WGAN loss
        use_gp : bool = True,    # Use gradient penalty
        use_plr : bool = True,   # Use path length regularization
        style_mixing_prob : float = 0.9,
        gp_weight : float = 10.0,    # Weight for gradient penalty
        pl_weight : float = 2.0,     # Weight for path length regularization
        pl_beta : float = 0.99,      # Exponential moving average coefficient to use for path length regularization,
        disc_optim_steps : int = 1,  # Number of steps to perform for the discriminator before stepping for the generator
        lazy_reg_steps_generator : int = 8,     # Lazy regularization steps for generator, 0 for no regularization
        lazy_reg_steps_discriminator: int = 16, # Lazy regularization steps for discriminator, 0 for no regularization
        learning_rate : float = 2e-3,
        adam_beta1 : float = 0,
        adam_beta2 : float = 0.99,
        adam_eps : float = 1e-8,
        save_every : int = 1000,
        gen_ema_beta : float = 0.999
    ):
        self.root_path = root_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_steps = total_steps
        self.device = device

        assert loss_type in ["vanilla", "wgan"]
        self.loss_type = loss_type

        if self.loss_type == "vanilla":
            self.D_loss = losses.VanillaDiscriminatorLoss()
            self.G_loss = losses.VanillaGeneratorLossNS()
        
        else:
            self.D_loss = losses.WGANDiscriminatorLoss()
            self.G_loss = losses.WGANGeneratorLoss()
    
        self.use_gp = use_gp
        self.use_plr = use_plr
        self.style_mixing_prob = style_mixing_prob

        if self.use_gp:
            self.D_gp = losses.GradientPenalty(reg_weight = gp_weight, gp_type = "r1" if self.loss_type == "vanilla" else "wgan-gp")

        if self.use_plr:
            self.G_plr = losses.PathLengthPenalty(reg_weight = pl_weight, beta = pl_beta)

        self.save_every = save_every
        self.gen_ema_beta = gen_ema_beta
        self.disc_optim_steps = disc_optim_steps
        self.lazy_reg_steps_generator = lazy_reg_steps_generator
        self.lazy_reg_steps_discriminator = lazy_reg_steps_discriminator
        self.next_d_step = 16
        self.next_g_step = 1
        # Terminate when self.next_g_step > self.total_steps
    
        # TODO: Cover cases when loading from checkpoints
        if checkpoint_path is not None:
            pass

        elif load_latest is not None:
            pass

        elif load_best_fid_val is not None:
            pass
        
        else:
            self.MN = MN.to(device)
            self.G = G.to(device)
            self.D = D.to(device)

            self.MNE = copy.deepcopy(self.MN).eval()
            self.GE = copy.deepcopy(self.G).eval()

            # Turn off gradient computation for generator EMWA
            requires_grad(self.MNE, False)
            requires_grad(self.GE, False)

            # Initialize all parameters of exponential moving average models to 0
            # TODO: When implementing truncation trick, don't forget to copy w_avg buffer into EMA w_avg buffer without applying EMA update rule!
            for param in self.MNE.parameters():
                param.data = torch.zeros_like(param.data)

            for param in self.GE.parameters():
                param.data = torch.zeros_like(param.data)

            g_scale = (self.lazy_reg_steps_generator) / (self.lazy_reg_steps_generator + 1)
            d_scale = (self.lazy_reg_steps_discriminator) / (self.lazy_reg_steps_discriminator + 1)

            # Scaling of optimizer hyperparameters to account for lazy regularization. Mentioned in StyleGAN2 paper as well, in appendix B under lazy regularziation
            # Reference code from StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L201
            self.optim_MN = torch.optim.Adam(MN.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
            self.optim_G = torch.optim.Adam(G.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
            self.optim_D = torch.optim.Adam(D.parameters(), lr = learning_rate * d_scale, betas = (adam_beta1 ** d_scale, adam_beta2 ** d_scale), eps = adam_eps)

    def discriminator_step(self, real_samples : torch.Tensor) -> None:
        start = time.time()
        requires_grad(self.D, True)
        requires_grad(self.G, False)
        requires_grad(self.MN, False)
        b, c, h, w = real_samples.shape
        device = real_samples.get_device()
        should_reg = self.lazy_reg_steps_discriminator > 0 and self.next_d_step % self.lazy_reg_steps_discriminator == 0

        # Potentially required for gradient penalty
        real_samples.requires_grad_(should_reg)
        real_pred = self.D(real_samples)
        w, _, _ = generate_style_mixes(self.MN, h, b, device)
        fake_samples = self.G(w, generate_noise(h, b, device))
        fake_pred = self.D(fake_samples)

        d_loss = self.D_loss(real_pred, fake_pred)

        if should_reg:
            # Use R1 gradient penalty for vanilla GAN loss            
            if self.loss_type == "vanilla":
                print(real_pred)
                d_loss += self.D_gp(real_samples, real_pred) 

            # For WGAN loss we use GP penalty, norm deviations from 1 and fake/real interpolation
            else:
                w, _, _ = generate_style_mixes(self.MN, h, b, device)
                fake_samples = self.G(w, generate_noise(h, b, device))
                d_loss += self.D_gp(real_samples, real_pred = None, fake_samples = fake_samples, critic = self.D)

            real_samples.requires_grad_(False)

        d_loss.backward()
        self.optim_D.step()
        self.optim_D.zero_grad()
        self.next_d_step += 1

        print(f"Time taken for discriminator step: {time.time() - start:.4f}s.")

    def __generator_step(self):
        pass
    
    def __ema_step(self):
        """
        Perform exponential moving average update of Generator weights (including the mapping network), using gen_ema_coeff. 
        This was first introduced, to my knowledge at least in ProGAN paper, https://arxiv.org/pdf/1710.10196. Another paper goes into mathematical details
        of why this works: https://arxiv.org/pdf/1806.04498

        Based on StyleGAN-2 implementation, they use exponential decay based on hyperparameters that can be found in 
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L297. However, for default values, this defaults to exponential moving
        average decay of approximately 0.999, which is the same as in ProGAN paper.
        """
        for param, ema_param in zip(self.MN.parameters(), self.MNE.parameters()):
            ema_param.data = torch.lerp(param.data, ema_param.data, self.gen_ema_beta)

        for param, ema_param in zip(self.G.parameters(), self.GE.parameters()):
            ema_param.data = torch.lerp(param.data, ema_param.data, self.gen_ema_beta)
        
        for buffer, ema_buffer in zip(self.MN.buffers(), self.MNE.buffers()):
            ema_buffer.copy_(buffer)
        
        for buffer, ema_buffer in zip(self.G.buffers(), self.GE.buffers()):
            ema_buffer.copy_(buffer)

def cycle_data_loader(dl):
    while True:
        for sample in dl:
            yield sample

if __name__ == "__main__":
    dataset = Dataset("celeba_128_v2", alpha = 1)
    dl = get_data_loader(dataset, 32)
    dl = cycle_data_loader(dl)

    len_dl = math.ceil(len(dataset) / 32)
    iters = []

    print(len_dl)

    for i in range(2 * len_dl):
        iters.append(next(dl))
    
    print(len(iters))

    for i in range(0, len_dl):
        print(torch.allclose(iters[i], iters[i + len_dl]))