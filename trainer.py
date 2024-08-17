import torch
import os
import copy
import losses
import math
import tqdm
import numpy as np

from model import Generator, Discriminator, MappingNetwork
from dataset import Dataset, get_data_loader
from typing import Union, Tuple
from utils import generate_style_mixes, generate_noise, generate_samples
from torchvision.utils import make_grid
from PIL import Image

# For now, supports only single GPU training. Also, no gradient accumulation, for now.
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
        target_resolution : int,
        batch_size : int,
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
        gen_ema_beta : float = 0.999,
        grad_accum_steps : int = 1,
        compute_generator_ema : bool = True
    ):
        self.root_path = root_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.total_steps = total_steps
        self.device = device
        self.target_resolution = target_resolution
        self.batch_size = batch_size
        
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
        self.compute_generator_ema = compute_generator_ema
        self.grad_accum_steps = grad_accum_steps

        self.disc_optim_steps = disc_optim_steps
        self.lazy_reg_steps_generator = lazy_reg_steps_generator
        self.lazy_reg_steps_discriminator = lazy_reg_steps_discriminator
        self.d_steps = 0
        self.g_steps = 0
        # Terminate when self.g_steps == self.total_steps
    
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

            self.MNE = copy.deepcopy(self.MN).to(device).eval()
            self.GE = copy.deepcopy(self.G).to(device).eval()
                
            g_scale = (self.lazy_reg_steps_generator) / (self.lazy_reg_steps_generator + 1)
            d_scale = (self.lazy_reg_steps_discriminator) / (self.lazy_reg_steps_discriminator + 1)

            # Scaling of optimizer hyperparameters to account for lazy regularization. Mentioned in StyleGAN2 paper as well, in appendix B under lazy regularziation
            # Reference code from StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L201
            self.optim_MN = torch.optim.Adam(MN.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
            self.optim_G = torch.optim.Adam(G.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
            self.optim_D = torch.optim.Adam(D.parameters(), lr = learning_rate * d_scale, betas = (adam_beta1 ** d_scale, adam_beta2 ** d_scale), eps = adam_eps)

            if not os.path.exists(self.root_path):
                os.mkdir(self.root_path)

    # Make private once initial tests are done
    def discriminator_step(self) -> Tuple[float, float]:
        self.optim_D.zero_grad()
        should_reg = self.lazy_reg_steps_discriminator > 0 and (self.d_steps + 1) % self.lazy_reg_steps_discriminator == 0

        d_loss, gp_loss = 0, 0

        for _ in range(self.grad_accum_steps):
            real_samples = next(self.train_loader).to(self.device)
            # Potential gradient penalty computation
            real_samples.requires_grad_(should_reg)

            real_pred = self.D(real_samples)
            ws, _, _ = generate_style_mixes(self.MN, self.target_resolution, self.batch_size, self.device, style_mixing_prob = self.style_mixing_prob)
            ws = ws.detach() # Detach mapping network from computational graph
            fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device)).detach() # Detach generator from computational graph
            fake_pred = self.D(fake_samples)

            current_d_loss = self.D_loss(real_pred, fake_pred) / self.grad_accum_steps
            current_gp_loss = 0

            d_loss += current_d_loss.item()

            if should_reg:
                # Use R1 gradient penalty for vanilla GAN loss            
                if self.loss_type == "vanilla":
                    current_gp_loss = self.D_gp(real_samples, real_pred) / self.grad_accum_steps
                    gp_loss += current_gp_loss.detach()

                # For WGAN loss we use GP penalty, norm deviations from 1 and fake/real interpolation
                else:
                    ws, _, _ = generate_style_mixes(self.MN, self.target_resolution, self.batch_size, self.device)
                    fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
                    current_gp_loss = self.D_gp(real_samples, real_pred = None, fake_samples = fake_samples, critic = self.D) / self.grad_accum_steps
                    gp_loss += current_gp_loss.detach()
                
            
            current_loss = current_d_loss + current_gp_loss
            current_loss.backward()

        self.optim_D.step()
        self.d_steps += 1
        return d_loss, gp_loss

    # Make private once initial tests are done
    def generator_step(self) -> Tuple[float, float]:
        self.optim_MN.zero_grad()
        self.optim_G.zero_grad()

        should_reg = self.lazy_reg_steps_generator > 0 and (self.g_steps + 1) % self.lazy_reg_steps_generator == 0

        g_loss, plr_loss = 0, 0

        for _ in range(self.grad_accum_steps):
            ws, _, _ = generate_style_mixes(self.MN, self.target_resolution, self.batch_size, self.device, style_mixing_prob = self.style_mixing_prob)
            fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
            fake_pred = self.D(fake_samples)

            current_g_loss = self.G_loss(fake_pred) / self.grad_accum_steps
            current_plr_loss = 0

            g_loss += current_g_loss.detach()

            # Apparently path length regularization uses new fake samples from generator?
            # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py#L77
            if should_reg:
                ws, _, _ = generate_style_mixes(self.MN, self.target_resolution, self.batch_size, self.device, style_mixing_prob = self.style_mixing_prob)
                fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
                current_plr_loss = self.G_plr(ws, fake_samples)  / self.grad_accum_steps
                plr_loss += current_plr_loss.detach()
            
            current_loss = current_g_loss + current_plr_loss
            current_loss.backward()

        self.optim_MN.step()
        self.optim_G.step()
        self.g_steps += 1
        return g_loss, plr_loss
    
    def __ema_generator_step(self):
        """
        Perform exponential moving average update of Generator weights (including the mapping network), using gen_ema_coeff. 
        This was first introduced, to my knowledge at least in ProGAN paper, https://arxiv.org/pdf/1710.10196. Another paper goes into mathematical details
        of why this works: https://arxiv.org/pdf/1806.04498

        Based on StyleGAN-2 implementation, they use exponential decay based on hyperparameters that can be found in 
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L297. However, for default values, this defaults to exponential moving
        average decay of approximately 0.999, which is the same as in ProGAN paper.
        """
        with torch.no_grad():
            for param, ema_param in zip(self.MN.parameters(), self.MNE.parameters()):
                if param.requires_grad:
                    ema_param = torch.lerp(param, ema_param, self.gen_ema_beta)
                
                else:
                    ema_param.copy_(param)

            for param, ema_param in zip(self.G.parameters(), self.GE.parameters()):
                if param.requires_grad:
                    ema_param = torch.lerp(param, ema_param, self.gen_ema_beta)
                
                else:
                    ema_param.copy_(param)

    def train(self, truncation_psi_inference = 0.7, style_mixing_prob_inference = 0.9, num_images_inference : int = 16, num_generated_rows : int = 1):
        
        pbar = tqdm.tqdm(range(self.g_steps, self.total_steps), position = 0, leave = True)
        for i in pbar:
            for j in range(self.disc_optim_steps):
                l, r = self.discriminator_step()
                pbar.write(f"Discriminator step: {self.d_steps}. Network loss: {l}. Regularization loss: {r}")

            l, r = self.generator_step()
            pbar.write(f"Generator step: {self.g_steps}. Network loss: {l}. Regularization loss: {r}")

            if self.compute_generator_ema:
                self.__ema_generator_step()
            
            if self.g_steps % self.save_every == 0:
                # Inference only for now
                with torch.no_grad():
                    fake_samples = generate_samples(self.GE, self.MNE, self.target_resolution, num_images_inference, self.device,
                                                    style_mixing_prob = style_mixing_prob_inference,
                                                    truncation_psi = truncation_psi_inference,
                                                    update_w_ema = True,
                                                    num_generated_rows = num_generated_rows)
                    
                    Image.fromarray(fake_samples, mode = "RGB").save(os.path.join(self.root_path, f"output_{self.g_steps}.jpg"))

if __name__ == "__main__":
    DEVICE = "cuda"
    target_res = 128
    mn = MappingNetwork(512, 8).to(DEVICE)
    g = Generator(128, 512, use_tanh_last = True).to(DEVICE)
    d = Discriminator(128, 3).to(DEVICE)

    num_images = 1e5
    steps = math.ceil(num_images // 128)

    dataset = Dataset("celeba_128_v2")
    dl = get_data_loader(dataset, 32)

    t = Trainer(mn, g, d, "first_model", dl, dl, steps, DEVICE, 128, 32, loss_type = "vanilla", save_every = 100, learning_rate = 2e-3, grad_accum_steps = 4)
    t.train(num_generated_rows = 4)
    # samples = generate_samples(g, mn, 128, 16, DEVICE, style_mixing_prob = 0, num_generated_rows = 4)
    # Image.fromarray(samples, mode = "RGB").show()