import torch
import os
import copy
import losses
import json
import tqdm
import shutil
import argparse
import csv

from model import Generator, Discriminator, MappingNetwork
from dataset import Dataset, get_data_loader
from typing import Union, Type, List
from utils import generate_noise, generate_samples, style_mixing, find_label
from PIL import Image
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
from contextlib import contextmanager

@contextmanager
def maybe_no_sync(module, condition):
    if isinstance(module, DistributedDataParallel) and condition:
        with module.no_sync():
            yield
        
    else:
        yield

class Trainer:
    def __init__(self,
        MN : Union[MappingNetwork, DistributedDataParallel], # Mapping network component of generator, None if running from pretrained
        G : Union[Generator, DistributedDataParallel], # Synthesis network component of generator, None if running from pretrained
        D: Union[Discriminator, DistributedDataParallel], # Discriminator, None if running from pretrained
        root_path : str,
        target_resolution : int,
        batch_size : int,
        loss_type : str = "vanilla",    # Use vanilla GAN loss or WGAN loss
        use_gp : bool = True,    # Use gradient penalty
        use_plr : bool = True,   # Use path length regularization
        style_mixing_prob : float = 0.9,
        style_mixing_prob_inference : List[float] = [0],
        truncation_psi_inference : List[float] = [0.2],
        gp_weight : float = 10.0,    # Weight for gradient penalty
        pl_weight : float = 2.0,     # Weight for path length regularization
        pl_beta : float = 0.99,      # Exponential moving average coefficient to use for path length regularization,
        disc_optim_steps : int = 1,  # Number of steps to perform for the discriminator before stepping for the generator
        lazy_reg_steps_generator : int = 8,     # Lazy regularization steps for generator, 0 for no regularization
        lazy_reg_steps_discriminator: int = 16, # Lazy regularization steps for discriminator, 0 for no regularization
        learning_rate : float = 2e-3,
        adam_beta1 : float = 0.0,
        adam_beta2 : float = 0.99,
        adam_eps : float = 1e-8,
        save_every : int = 1000,    # Create checkpoint every save_every generator steps. Also corresponds to evaluation strategy
        gen_ema_beta : float = 0.999,
        grad_accum_steps : int = 1,
        compute_generator_ema : bool = True,
        ema_steps_threshold : int = 1, # Compute Generator EMA starting from ema_steps_threshold,
        w_estimate_samples : int = 20000,
        save_total_limit : int = 1,
        num_images_inference : int = 16,
        num_generated_rows : int = 1,
        sample_every : int = 1
    ):            
        self.root_path = root_path
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
        self.style_mixing_prob_inference = style_mixing_prob_inference
        self.truncation_psi_inference = truncation_psi_inference
        self.gp_weight = gp_weight
        self.pl_weight = pl_weight
        self.pl_beta = pl_beta

        if self.use_gp:
            self.D_gp = losses.GradientPenalty(reg_weight = gp_weight, gp_type = "r1" if self.loss_type == "vanilla" else "wgan-gp")

        if self.use_plr:
            self.G_plr = losses.PathLengthPenalty(reg_weight = pl_weight, beta = pl_beta)

        self.save_every = save_every
        self.gen_ema_beta = gen_ema_beta
        self.compute_generator_ema = compute_generator_ema
        self.ema_steps_threshold = ema_steps_threshold
        self.w_estimate_samples = w_estimate_samples
        self.save_total_limit = save_total_limit
        self.num_images_inference = num_images_inference
        self.num_generated_rows = num_generated_rows
        self.sample_every = sample_every
        self.save_history = []

        self.grad_accum_steps = grad_accum_steps
        self.disc_optim_steps = disc_optim_steps
        self.lazy_reg_steps_generator = lazy_reg_steps_generator
        self.lazy_reg_steps_discriminator = lazy_reg_steps_discriminator
        self.d_steps = 0
        self.g_steps = 0

        self.MN = MN
        self.G = G
        self.D = D
        
        self.MNE = None
        self.GE = None
            
        g_scale = (self.lazy_reg_steps_generator) / (self.lazy_reg_steps_generator + 1)
        d_scale = (self.lazy_reg_steps_discriminator) / (self.lazy_reg_steps_discriminator + 1)

        # Incremental averages of discriminator loss, discriminator gradient penalty, generator loss and generator path length regularization, respectivley
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        # Scaling of optimizer hyperparameters to account for lazy regularization. Mentioned in StyleGAN2 paper as well, in appendix B under lazy regularziation
        # Reference code from StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L201
        self.optim_MN = torch.optim.Adam(MN.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
        self.optim_G = torch.optim.Adam(G.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
        self.optim_D = torch.optim.Adam(D.parameters(), lr = learning_rate * d_scale, betas = (adam_beta1 ** d_scale, adam_beta2 ** d_scale), eps = adam_eps)
    
    def __create_csv(self, name, header):
        if not os.path.exists(os.path.join(self.root_path, name)):
            with open(os.path.join(self.root_path, name), "w+", encoding = "utf-8") as f:
                writer = csv.writer(f, lineterminator = "\n")
                writer.writerow([header])
    
    def __write_csv(self, name, data):
        with open(os.path.join(self.root_path, name), "a+", encoding = "utf-8") as f:
            writer = csv.writer(f, lineterminator = "\n")
            writer.writerow([data])
    
    def __get_raw_model(self, model):
        return model if not isinstance(model, DistributedDataParallel) else model.module
    
    @classmethod
    def from_trained(cls : Type["Trainer"], root_path : str, local_rank : int, device : str, is_ddp : bool, is_master_process : bool):
        '''
        Load latest, load specific checkpoint or load checkpoint that obtained best FID so far
        '''
        def from_checkpoint(root, checkpoint) -> Type["Trainer"]:
            checkpoint_path = os.path.join(root, checkpoint)

            with open(os.path.join(root, ".metadata.json"), "r", encoding = "utf-8") as f:
                global_metadata = json.load(f)
            
            with open(os.path.join(checkpoint_path, ".metadata.json"), "r", encoding = "utf-8") as f:
                local_metadata = json.load(f)

            MN_d = torch.load(os.path.join(checkpoint_path, "MN.pth"), weights_only = True)
            G_d = torch.load(os.path.join(checkpoint_path, "G.pth"), weights_only = True)
            D_d = torch.load(os.path.join(checkpoint_path, "D.pth"), weights_only = True)

            MN = MappingNetwork.from_dict(MN_d).to(device)
            G = Generator.from_dict(G_d).to(device)
            D = Discriminator.from_dict(D_d).to(device)

            if is_ddp:
                MN = DistributedDataParallel(MN, device_ids = [local_rank])
                G = DistributedDataParallel(G, device_ids = [local_rank])
                D = DistributedDataParallel(D, device_ids = [local_rank])

            mne_path = os.path.join(checkpoint_path, "MNE.pth")
            MNE, GE = None, None

            if os.path.exists(mne_path) and is_master_process:
                MN_d = torch.load(os.path.join(checkpoint_path, "MNE.pth"), weights_only = True)
                G_d = torch.load(os.path.join(checkpoint_path, "GE.pth"), weights_only = True)
                MNE = MappingNetwork.from_dict(MN_d).to(device)
                GE = Generator.from_dict(G_d).to(device)

            kwargs_metadata = copy.deepcopy(global_metadata)
            kwargs_metadata.pop("target_resolution")
            kwargs_metadata.pop("mapping_network_params")
            kwargs_metadata.pop("generator_params")
            kwargs_metadata.pop("discriminator_params")
            kwargs_metadata.pop("batch_size")

            res = Trainer(MN, G, D, 
                          root,
                          global_metadata["target_resolution"],
                          global_metadata["batch_size"],
                          **kwargs_metadata
                          )
            
            res.d_steps = local_metadata["d_steps"]
            res.g_steps = local_metadata["g_steps"]
            res.save_history = local_metadata["save_history"]

            res.optim_MN.load_state_dict(torch.load(os.path.join(checkpoint_path, "MN_optimizer.pth"), weights_only = True))
            res.optim_G.load_state_dict(torch.load(os.path.join(checkpoint_path, "G_optimizer.pth"), weights_only = True))
            res.optim_D.load_state_dict(torch.load(os.path.join(checkpoint_path, "D_optimizer.pth"), weights_only = True))

            if MNE is not None:
                res.MNE = MNE
                res.GE = GE

            return res
                
        target_label = find_label(root_path, "g_steps", lambda x, y : x > y)
        assert target_label is not None, f"No valid model checkpoint fount in {root_path}"
        return from_checkpoint(root_path, target_label)
                
    def __global_metadata_dict(self) -> dict:
        d = {
            "target_resolution": self.target_resolution,
            "loss_type": self.loss_type,
            "use_gp": self.use_gp,
            "use_plr": self.use_plr,
            "style_mixing_prob": self.style_mixing_prob,
            "style_mixing_prob_inference": self.style_mixing_prob_inference,
            "truncation_psi_inference": self.truncation_psi_inference,
            "gp_weight": self.gp_weight,
            "pl_weight": self.pl_weight,
            "pl_beta": self.pl_beta,
            "disc_optim_steps": self.disc_optim_steps,
            "lazy_reg_steps_generator": self.lazy_reg_steps_generator,
            "lazy_reg_steps_discriminator": self.lazy_reg_steps_discriminator,
            "learning_rate": self.learning_rate,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "save_every": self.save_every,
            "gen_ema_beta": self.gen_ema_beta,
            "grad_accum_steps": self.grad_accum_steps,
            "compute_generator_ema": self.compute_generator_ema,
            "ema_steps_threshold": self.ema_steps_threshold,
            "w_estimate_samples": self.w_estimate_samples,
            "mapping_network_params": self.__get_raw_model(self.MN).to_dict(),
            "generator_params": self.__get_raw_model(self.G).to_dict(),
            "discriminator_params": self.__get_raw_model(self.D).to_dict(),
            "save_total_limit": self.save_total_limit,
            "batch_size": self.batch_size,
            "sample_every": self.sample_every,
            "num_images_inference": self.num_images_inference,
            "num_generated_rows": self.num_generated_rows
        }

        return d

    def __local_metadata_dict(self) -> dict:
        d = {
            "d_steps": self.d_steps,
            "g_steps": self.g_steps,
            "save_history": self.save_history
        }

        return d
    
    def __create_checkpoint(self, check_name : str) -> None:
        path = os.path.join(self.root_path, check_name)

        if os.path.exists(path):
            # Re-create checkpoint if it already existed, will delete existing data.
            shutil.rmtree(path)

        os.mkdir(path)

        with open(os.path.join(path, ".metadata.json"), "w+", encoding = "utf-8") as f:
            json.dump(self.__local_metadata_dict(), f, indent = 4)
        
        torch.save(self.__get_raw_model(self.MN).to_dict(state_dict = True), os.path.join(path, "MN.pth"))
        torch.save(self.__get_raw_model(self.G).to_dict(state_dict = True), os.path.join(path, "G.pth"))
        torch.save(self.__get_raw_model(self.D).to_dict(state_dict = True), os.path.join(path, "D.pth"))
        torch.save(self.optim_MN.state_dict(), os.path.join(path, "MN_optimizer.pth"))
        torch.save(self.optim_G.state_dict(), os.path.join(path, "G_optimizer.pth"))
        torch.save(self.optim_D.state_dict(), os.path.join(path, "D_optimizer.pth"))
        
        if self.MNE is not None:
            torch.save(self.__get_raw_model(self.MNE).to_dict(state_dict = True), os.path.join(path, "MNE.pth"))
            torch.save(self.__get_raw_model(self.G).to_dict(state_dict = True), os.path.join(path, "GE.pth"))
        
    def __discriminator_step(self, world_size : int, batch_size : int, train_loader : torch.utils.data.DataLoader, device : str):
        self.optim_D.zero_grad()
        should_reg = self.lazy_reg_steps_discriminator > 0 and (self.d_steps + 1) % self.lazy_reg_steps_discriminator == 0 and self.use_gp

        d_loss, gp_loss = 0, 0

        for i in range(self.grad_accum_steps):
            with maybe_no_sync(self.D, i == self.grad_accum_steps - 1):
                real_samples = next(train_loader).to(device)
                # Potential gradient penalty computation
                real_samples.requires_grad_(should_reg)

                real_pred = self.D(real_samples)
                ws, _, _ = style_mixing(self.MN, self.__get_raw_model(self.G).num_layers * 2, batch_size, device, self.style_mixing_prob)
                fake_samples = self.G(ws, generate_noise(self.target_resolution, batch_size, device)).detach() # Detach generator from computational graph
                fake_pred = self.D(fake_samples)

                current_d_loss = self.D_loss(real_pred, fake_pred) / self.grad_accum_steps
                current_gp_loss = 0

                d_loss += current_d_loss.item()

                if should_reg:
                    # Use R1 gradient penalty for vanilla GAN loss  
                    # Quote from StyleGAN2 paper: "We also multiply the regularization term by k to balance the overall magnitude of its gradients"          
                    if self.loss_type == "vanilla":
                        current_gp_loss = (self.D_gp(real_samples, real_pred) / self.grad_accum_steps) * self.lazy_reg_steps_discriminator

                    # For WGAN loss we use GP penalty, norm deviations from 1 and fake/real interpolation
                    else:
                        ws, _, _ = style_mixing(self.MN, self.__get_raw_model(self.G).num_layers * 2, batch_size, device, self.style_mixing_prob)
                        fake_samples = self.G(ws, generate_noise(self.target_resolution, batch_size, device))
                        current_gp_loss = (self.D_gp(real_samples, real_pred = None, fake_samples = fake_samples, critic = self.D) / self.grad_accum_steps) * self.lazy_reg_steps_discriminator
                    
                    gp_loss += current_gp_loss.item()
                
                # Account for world_size when doing backward
                current_loss = (current_d_loss + current_gp_loss) / world_size
                current_loss.backward()

        self.optim_D.step()
        self.d_steps += 1

        return d_loss, gp_loss

    # Make private once initial tests are done
    def __generator_step(self, world_size : int, batch_size : int, device : str):
        self.optim_MN.zero_grad()
        self.optim_G.zero_grad()

        should_reg = self.lazy_reg_steps_generator > 0 and (self.g_steps + 1) % self.lazy_reg_steps_generator == 0 and self.use_plr

        g_loss, plr_loss = 0, 0

        for i in range(self.grad_accum_steps):
            with maybe_no_sync(self.G, i == self.grad_accum_steps - 1), maybe_no_sync(self.MN, i == self.grad_accum_steps - 1):
                ws, _, _ = style_mixing(self.MN, self.__get_raw_model(self.G).num_layers * 2, batch_size, device, self.style_mixing_prob)
                fake_samples = self.G(ws, generate_noise(self.target_resolution, batch_size, device))
                fake_pred = self.D(fake_samples)

                current_g_loss = self.G_loss(fake_pred) / self.grad_accum_steps
                current_plr_loss = 0

                g_loss += current_g_loss.item()

                # Apparently path length regularization uses new fake samples from generator
                # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py#L77
                if should_reg:
                    ws, _, _ = style_mixing(self.MN, self.__get_raw_model(self.G).num_layers * 2, batch_size, device, self.style_mixing_prob)
                    fake_samples = self.G(ws, generate_noise(self.target_resolution, batch_size, device))
                    current_plr_loss = (self.G_plr(ws, fake_samples)  / self.grad_accum_steps) * self.lazy_reg_steps_generator
                    plr_loss += current_plr_loss.item()
                
                current_loss = (current_g_loss + current_plr_loss) / world_size
                current_loss.backward()

        self.optim_MN.step()
        self.optim_G.step()
        self.g_steps += 1
        return g_loss, plr_loss
    

    def __ema_generator_step(self, device : str):
        """
        Perform exponential moving average update of Generator weights (including the mapping network), using gen_ema_coeff. 
        This was first introduced, to my knowledge at least in ProGAN paper, https://arxiv.org/pdf/1710.10196. Another paper goes into mathematical details
        of why this works: https://arxiv.org/pdf/1806.04498

        Based on StyleGAN-2 implementation, they use exponential decay based on hyperparameters that can be found in 
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L297. However, for default values, this defaults to exponential moving
        average decay of approximately 0.999, which is the same as in ProGAN paper.
        """

        if not (self.compute_generator_ema and self.g_steps >= self.ema_steps_threshold):
            return
        
        if self.GE is None:
            self.GE = copy.deepcopy(self.G).to(device).eval()
            self.MNE = copy.deepcopy(self.MN).to(device).eval()
            return
        
        with torch.no_grad():
            for param, ema_param in zip(self.MN.parameters(), self.MNE.parameters()):
                if param.requires_grad:
                    ema_param.data = torch.lerp(param.data, ema_param.data, self.gen_ema_beta)
                
                else:
                    ema_param.copy_(param)

            for param, ema_param in zip(self.G.parameters(), self.GE.parameters()):
                if param.requires_grad:
                    ema_param.data = torch.lerp(param.data, ema_param.data, self.gen_ema_beta)
                
                else:
                    ema_param.copy_(param)

    def __model_step(self, world_size : int, batch_size : int, pbar : tqdm.tqdm, train_loader : torch.utils.data.DataLoader, device : str, is_local_master : bool):
        disc_losses, disc_regs = [], []

        for _ in range(self.disc_optim_steps):
            l, r = self.__discriminator_step(world_size, batch_size, train_loader, device)
            disc_losses.append(l)
            
            # If previous iteration was for regularization
            if self.d_steps % self.lazy_reg_steps_discriminator == 0:
                disc_regs.append(r)
            
        l, r = self.__generator_step(world_size, batch_size, device)

        if is_local_master:
            pbar.set_postfix({"G loss": l, 
                            "G PLR": r, 
                            "D loss avg": sum(disc_losses) / len(disc_losses), 
                            "D GP avg": sum(disc_regs) / len(disc_regs) if len(disc_regs) > 0 else 0})

            # Local masters should write loss data to non-shared filesystems so that it can be accumulated
            for d_loss in disc_losses:
                self.__write_csv("d_adversarial_loss.csv", d_loss)
            
            for d_reg in disc_regs:
                self.__write_csv("d_gp.csv", d_reg)
            
            self.__write_csv("g_adversarial_loss.csv", l)

            if self.g_steps % self.lazy_reg_steps_generator == 0:
                self.__write_csv("g_plr.csv", r)
            
    def train(self, 
              total_steps : int, 
              train_loader : torch.utils.data.DataLoader, 
              is_master : bool,
              is_local_master : bool,
              device : str,
              world_size : int):
        
        if is_master:
            if not os.path.exists(self.root_path):
                os.mkdir(self.root_path)

            # Seperate directory for storing samples obtained during training
            if not os.path.exists(os.path.join(self.root_path, "samples")):
                os.mkdir(os.path.join(self.root_path, "samples"))

            with open(os.path.join(self.root_path, ".metadata.json"), "w+", encoding = "utf-8") as f:
                json.dump(self.__global_metadata_dict(), f, indent = 4)
        
            self.__create_csv("d_adversarial_loss.csv", "adversarial_loss")
            self.__create_csv("g_adversarial_loss.csv", "adversarial_loss")

            if self.use_gp:
                self.__create_csv("d_gp.csv", "gradient_penalty")
            
            if self.use_plr:
                self.__create_csv("g_plr.csv", "path_length_regularization")

        pbar = None
        if is_local_master:
            pbar = tqdm.tqdm(total = total_steps, initial = self.g_steps, position = 0, leave = True)

        for _ in range(self.g_steps, total_steps):
            self.__model_step(world_size, self.batch_size, pbar, train_loader, device, is_local_master)

            if not is_master:
                continue

            self.__ema_generator_step(device)
            
            if self.g_steps % self.save_every == 0 or self.g_steps == total_steps:
                if self.save_total_limit > 0:
                    if self.save_total_limit == len(self.save_history):
                        target_steps = self.save_history[0]
                        target_label = find_label(self.root_path, "g_steps", lambda x, y : x == target_steps, assume_first = False)

                        if target_label is not None:
                            shutil.rmtree(os.path.join(self.root_path, target_label))

                        self.save_history.pop(0)

                    self.save_history.append(self.g_steps)

                self.__create_checkpoint(f"checkpoint_{self.g_steps}")

            if self.g_steps % self.sample_every == 0 or self.g_steps == total_steps:
                with torch.no_grad():
                    for i, psi in enumerate(self.truncation_psi_inference):
                        for j, smp in enumerate(self.style_mixing_prob_inference):
                            # Possible that EMA models don't exist if current g_steps < ema_steps_threshold 
                            if self.MNE is not None:
                                ema_samples = generate_samples(self.GE, self.MNE, device, self.num_images_inference,
                                                                style_mixing_prob = smp,
                                                                truncation_psi = psi,
                                                                num_generated_rows = self.num_generated_rows,
                                                                w_estimate_samples = self.w_estimate_samples,
                                                                update_w_ema = False)
                                    
                                Image.fromarray(ema_samples, mode = "RGB").save(os.path.join(self.root_path, "samples", f"ema_samples_{self.g_steps}_{i}{j}.jpg"))

                            current_samples = generate_samples(self.G, self.MN, device, self.num_images_inference,
                                                            style_mixing_prob = smp,
                                                            truncation_psi = psi,
                                                            num_generated_rows = self.num_generated_rows,
                                                            w_estimate_samples = self.w_estimate_samples,
                                                            update_w_ema = False)
                                
                            Image.fromarray(current_samples, mode = "RGB").save(os.path.join(self.root_path, "samples", f"current_samples_{self.g_steps}_{i}{j}.jpg"))

            pbar.update(1)
        
        if pbar is not None:
            pbar.close()

def parse_args():
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    # Mandatory
    parser.add_argument("path_train", type = str, help = "Path to training set images")
    parser.add_argument("target_res", type = int, default = 128, help = "Width and height of images, should be a power of 2 consistent with provided datasets, for now")
    parser.add_argument("model_dir", type = str, help = "Directory where model checkpoints and infered samples should be saved")

    # Optional
    parser.add_argument("--latent_dim", type = int, default = 512, help = "Dimensionality of style vectors")
    parser.add_argument("--batch_size", type = int, default = 32, help = "Per device batch size")
    parser.add_argument("--grad_accum", type = int, default = 4, help = "Gradient accumulations steps to perform on each device before summing gradients")
    parser.add_argument("--mn_depth", type = int, default = 8, help = "Depth of the mapping network")
    parser.add_argument("--training_steps", type = int, default = 1000, help = "Number of training steps")
    parser.add_argument("--target_num_images", type = int, help = "Number of images to train the model on. Will override training-steps, if given")
    parser.add_argument("--loss_type", type = str, default = "vanilla", help = "GAN loss type to be used for the model. Can be vanilla or wgan")
    parser.add_argument("--save_every", type = int, default = 1000, help = "Creates a model checkpoint every save-every steps. Also period of inference")
    parser.add_argument("--learning_rate", type = float, default = 2e-3, help = "Learning rate for optimizers")
    parser.add_argument("--mlp_lr_mul", type = float, default = 0.01, help = "Reduces learning rate of mapping network by a factor of mlp-lr-mul")
    parser.add_argument("--style_mixing_prob", type = float, default = 0.9, help = "Style mixing probability to use during training")
    parser.add_argument("--gen_ema_beta", type = float, default = 0.999, help = "Decay coefficient for EMA of mapping network and generator weights")
    parser.add_argument("--ema_steps_threshold", type = int, default = 3000, help = "Compute EMA of mapping network and generator weights only after ema-steps-threshold training steps")
    parser.add_argument("--network_capacity", type = int, default = 8, help = "Multiplicative factor for number of filters in generator and discriminator. Number of features maps for generator layer that generates images of resolution 2^k is obtained as f(k) = min(max_filters, network_capacity * 2^(log_2(target_res) - k + 1)), and similarly for discriminator layer that processes resolution 2^k has h(k) = min(max_filters, network_capacity * 2^(k - 1)). ")
    parser.add_argument("--gen_use_tanh_last", type = bool, default = False, help = "Use tanh in the last layer of generator to keep images in [-1, 1]. StyleGAN2 paper does not use this in the last layer")
    parser.add_argument("--disc_use_mbstd", type = bool, default = True, help = "Use minibatch-std in last layer of discriminator")
    parser.add_argument("--style_mixing_probs_inference", type = float, nargs = "+", default = [0.0], help = "Different style mixing probabilities to try during inference, pass as a space-seperated list of floats")
    parser.add_argument("--truncation_psis_inference", type = float, nargs = "+", default = [1], help = "Different psi-s for truncation trick to use during inference, pass as a space-seperated list of floats")
    parser.add_argument("--fir_filter_sampling", nargs = "+", default = [1, 3, 3, 1], help = "Unnormalized FIR filter to use in upsampling/downsampling layers")
    parser.add_argument("--w_ema_beta", type = float, default = 0.995, help = "EMA coefficient to use when estimating mean style vector in mapping network during training")
    parser.add_argument("--max_filters", type = int, default = 512, help = "Maximum number of filters to use in convolutional layers of generator and discriminator")
    parser.add_argument("--mbstd_group_size", type = int, default = 4, help = "Minibatch standard deviation group size for discriminator, --batch-size should be divisible by this")
    parser.add_argument("--mbstd_num_channels", type = int, default = 1, help = "Minibatch standard deviation number of channels for discriminator, should divide number of channels at the output of discriminator, before applying minibatchstd and flatten")
    parser.add_argument("--images_in_channels", type = int, default = 3, help = "Number of channels generated images/inspected images should have, 3 for RGB, 4 for RGBA, 1 for grayscale, etc.")
    parser.add_argument("--ddp_backend", type = str, default = "nccl", help = "DDP backend to use. If training on CPU, you should use gloo. Supported backends, and DDP documentation is available at: https://pytorch.org/docs/stable/distributed.html. If running from Windows, you might want to run with gloo, as nccl is not supported (at least at the time of writhing these scripts)")
    parser.add_argument("--from_checkpoint", type = bool, nargs = "?", default = False, const = True, help = "Continue training from latest checkpoint in model_dir")
    parser.add_argument("--num_images_inference", type = int, default = 16, help = "Number of images to generate for inference")
    parser.add_argument("--num_rows_inference", type = int, default = 4, help = "Number of rows to present generated images during inference. Should divide num_images_inference")
    parser.add_argument("--disc_optim_steps", type = int, default = 1, help = "Number of optimizations steps for discriminator, before optimizing generator")
    parser.add_argument("--random_seed", type = int, default = 1337, help = "Random seed for reproducibility. In multi GPU regime, this will be offset by global rank of each GPU, so each GPU will end up with a different seed")
    parser.add_argument("--no_w_ema", type = bool, nargs = "?", default = False, const = True, help = "Should mapping network use EMA for estimating mean style vector or not")
    parser.add_argument("--w_estimate_samples", type = int, default = 20000, help = "Number of samples taken from multivariate standard Normal distribution to use to estimate average style vector for truncation, in case --no_w_ema is turned on")
    parser.add_argument("--save_total_limit", type = int, default = 1, help = "Specifies number of model checkpoints that should be rotated. Non-positive value results in no limits")
    parser.add_argument("--sample_every", type = int, default = 1, help = "Generate samples every sample_every steps during training. These samples are saved under sampled subdirectory in model_dir, and should represent samples generated by model at different stages of training")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        backend = args.ddp_backend
        init_process_group(backend = backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        is_master = ddp_rank == 0
        is_local_master = ddp_local_rank == 0
        seed_offset = ddp_rank # each process gets a different seed
                
    else:
        # if not ddp, we are running on a single gpu, and one process
        is_master = True
        is_local_master = True
        seed_offset = 0
        ddp_world_size = 1
        ddp_rank = 0
        ddp_local_rank = 0
        
    torch.manual_seed(args.random_seed + seed_offset)

    if args.from_checkpoint:
        t = Trainer.from_trained(args.model_dir, ddp_local_rank, device, ddp, is_master)
        training_steps = args.training_steps

        if args.target_num_images is not None:
            training_steps = int(args.target_num_images / (ddp_world_size * t.batch_size * args.grad_accum))

        train_dataset = Dataset(args.path_train)
        train_dl = get_data_loader(train_dataset, t.batch_size, ddp, pin_memory = "cuda" in device)

        if is_local_master:
            print(f"Training from latest checkpoint found in {args.model_dir}.\nTraining steps based on target_num_images and training_steps parameteres: {training_steps}")

        t.train(training_steps, train_dl, is_master, is_local_master, device, ddp_world_size)
        
    else:
        mn = MappingNetwork(args.latent_dim, args.mn_depth, 
                            lr_mul = args.mlp_lr_mul, 
                            estimate_w = not args.no_w_ema,
                            w_ema_beta = args.w_ema_beta).to(device)
        
        g = Generator(args.target_res, args.latent_dim, 
                    network_capacity = args.network_capacity, 
                    max_features = args.max_filters, 
                    fir_filter = args.fir_filter_sampling,
                    use_tanh_last = args.gen_use_tanh_last,
                    rgb_out_channels = args.images_in_channels).to(device)
        
        d = Discriminator(args.target_res, args.images_in_channels, 
                        network_capacity = args.network_capacity,
                        max_features = args.max_filters,
                        use_mbstd = args.disc_use_mbstd,
                        mbstd_group_size = args.mbstd_group_size,
                        mbstd_num_channels = args.mbstd_num_channels,
                        fir_filter = args.fir_filter_sampling).to(device)
        
        if ddp:
            mn = DistributedDataParallel(mn, device_ids = [ddp_local_rank])
            g = DistributedDataParallel(g, device_ids = [ddp_local_rank])
            d = DistributedDataParallel(d, device_ids = [ddp_local_rank])
        
        train_dataset = Dataset(args.path_train)

        train_dl = get_data_loader(train_dataset, args.batch_size, ddp, pin_memory = "cuda" in device)
        
        t = Trainer(mn, g, d, args.model_dir, args.target_res, args.batch_size,
                    loss_type = args.loss_type, 
                    save_every = args.save_every, 
                    learning_rate = args.learning_rate, 
                    grad_accum_steps = args.grad_accum, 
                    style_mixing_prob = args.style_mixing_prob, 
                    gen_ema_beta = args.gen_ema_beta, 
                    ema_steps_threshold = args.ema_steps_threshold,
                    style_mixing_prob_inference = args.style_mixing_probs_inference,
                    truncation_psi_inference = args.truncation_psis_inference,
                    disc_optim_steps = args.disc_optim_steps,
                    w_estimate_samples = args.w_estimate_samples,
                    save_total_limit = args.save_total_limit,
                    num_images_inference = args.num_images_inference,
                    num_generated_rows = args.num_rows_inference,
                    sample_every = args.sample_every)
        
        training_steps = args.training_steps

        if args.target_num_images is not None:
            training_steps = int(args.target_num_images / (ddp_world_size * args.batch_size * args.grad_accum))

        if is_local_master:
            print(f"Training freshly initialized model.\nTraining steps based on target_num_images and training_steps parameters: {training_steps}")
            
        t.train(training_steps, train_dl, is_master, is_local_master, device, ddp_world_size)

    if ddp:
        destroy_process_group()