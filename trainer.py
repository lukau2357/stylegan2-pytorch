import torch
import os
import copy
import losses
import json
import tqdm
import shutil

from model import Generator, Discriminator, MappingNetwork
from dataset import Dataset, get_data_loader
from typing import Union, Tuple, Type, List
from utils import generate_noise, generate_samples, new_style_mixing
from PIL import Image
from losses import FID

# For now, supports only single GPU training. Also, no gradient accumulation, for now.
class Trainer:
    def __init__(self,
        MN : Union[MappingNetwork, None], # Mapping network component of generator, None if running from pretrained
        G : Union[Generator, None], # Synthesis network component of generator, None if running from pretrained
        D: Union[Discriminator, None], # Discriminator, None if running from pretrained
        root_path : str,
        total_steps : int,  # Number of training steps (for generator),
        device : str,
        target_resolution : int,
        batch_size : int,
        loss_type : str = "vanilla",    # Use vanilla GAN loss or WGAN loss
        use_gp : bool = True,    # Use gradient penalty
        use_plr : bool = True,   # Use path length regularization
        style_mixing_prob : float = 0.9,
        style_mixing_prob_inference : List[float] = [0.9],
        truncation_psi_inference : List[float] = [0.7],
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
        log_every : int = 20, # Log loss information every log_every steps,
        save_total_limit : Union[int, None] = None
    ):            
        self.root_path = root_path
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
        self.log_every = log_every
        self.grad_accum_steps = grad_accum_steps

        self.disc_optim_steps = disc_optim_steps
        self.lazy_reg_steps_generator = lazy_reg_steps_generator
        self.lazy_reg_steps_discriminator = lazy_reg_steps_discriminator
        self.d_steps = 0
        self.g_steps = 0
        self.save_total_limit = save_total_limit
        self.save_history = []

        self.MN = MN.to(device)
        self.G = G.to(device)
        self.D = D.to(device)

        self.MNE = None
        self.GE = None
            
        g_scale = (self.lazy_reg_steps_generator) / (self.lazy_reg_steps_generator + 1)
        d_scale = (self.lazy_reg_steps_discriminator) / (self.lazy_reg_steps_discriminator + 1)

        # Incremental averages of discriminator loss, discriminator gradient penalty, generator loss and generator path length regularization, respectivley
        self.d_loss_ia = 0
        self.d_gp_ia = 0
        self.g_loss_ia = 0
        self.g_plr_ia = 0

        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps

        # Scaling of optimizer hyperparameters to account for lazy regularization. Mentioned in StyleGAN2 paper as well, in appendix B under lazy regularziation
        # Reference code from StyleGAN2-ADA: https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/training_loop.py#L201
        self.optim_MN = torch.optim.Adam(MN.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
        self.optim_G = torch.optim.Adam(G.parameters(), lr = learning_rate * g_scale, betas = (adam_beta1 ** g_scale, adam_beta2 ** g_scale), eps = adam_eps)
        self.optim_D = torch.optim.Adam(D.parameters(), lr = learning_rate * d_scale, betas = (adam_beta1 ** d_scale, adam_beta2 ** d_scale), eps = adam_eps)

        if not os.path.exists(self.root_path):
            os.mkdir(self.root_path)
        
        with open(os.path.join(self.root_path, ".metadata.json"), "w+", encoding = "utf-8") as f:
            json.dump(self.__global_metadata_dict(), f, indent = 4)

    @classmethod
    def from_trained(cls : Type["Trainer"], root_path : str, checkpoint_path : Union[str, None] = None, load_latest: Union[bool, None] = None, load_best_fid : Union[bool, None] = None):
        def from_checkpoint(root, checkpoint) -> Type["Trainer"]:
            checkpoint_path = os.path.join(root, checkpoint)

            with open(os.path.join(root, ".metadata.json"), "r", encoding = "utf-8") as f:
                global_metadata = json.load(f)
            
            with open(os.path.join(checkpoint_path, ".metadata.json"), "r", encoding = "utf-8") as f:
                local_metadata = json.load(f)

            MN = MappingNetwork.from_dict(global_metadata["mapping_network_params"])
            G = Generator.from_dict(global_metadata["generator_params"])
            D = Discriminator.from_dict(global_metadata["discriminator_params"])

            MN.load_state_dict(torch.load(os.path.join(checkpoint_path, "MN.pth"), weights_only = True))
            G.load_state_dict(torch.load(os.path.join(checkpoint_path, "G.pth"), weights_only = True))
            D.load_state_dict(torch.load(os.path.join(checkpoint_path, "D.pth"), weights_only = True))

            mne_path = os.path.join(checkpoint_path, "MNE.pth")
            MNE, GE = None, None

            if os.path.exists(mne_path):
                # TODO: device for Trainer is fixed, if it started training on CPU it would be "impossible" to continue training from GPU
                # fix this some day.
                MNE = MappingNetwork.from_dict(global_metadata["mapping_network_params"]).to(global_metadata["device"])
                GE = Generator.from_dict(global_metadata["generator_params"]).to(global_metadata["device"])
                MNE.load_state_dict(torch.load(mne_path, weights_only = True))
                GE.load_state_dict(torch.load(os.path.join(checkpoint_path, "GE.pth"), weights_only = True))

            kwargs_metadata = copy.deepcopy(global_metadata)
            kwargs_metadata.pop("total_steps")
            kwargs_metadata.pop("device")
            kwargs_metadata.pop("target_resolution")
            kwargs_metadata.pop("batch_size")
            kwargs_metadata.pop("mapping_network_params")
            kwargs_metadata.pop("generator_params")
            kwargs_metadata.pop("discriminator_params")

            res = Trainer(MN, G, D, 
                          root, 
                          global_metadata["total_steps"], 
                          global_metadata["device"], 
                          global_metadata["target_resolution"],
                          global_metadata["batch_size"],
                          **kwargs_metadata
                          )
            
            res.d_steps = local_metadata["d_steps"]
            res.g_steps = local_metadata["g_steps"]
            res.d_loss_ia = local_metadata["d_loss_ia"]
            res.d_gp_ia = local_metadata["d_gp_ia"]
            res.g_loss_ia = local_metadata["g_loss_ia"]
            res.g_plr_ia = local_metadata["g_plr_ia"]
            res.save_total_limit = local_metadata["save_total_limit"]
            res.save_history = local_metadata["save_history"]

            res.optim_MN.load_state_dict(torch.load(os.path.join(checkpoint_path, "MN_optimizer.pth"), weights_only = True))
            res.optim_G.load_state_dict(torch.load(os.path.join(checkpoint_path, "G_optimizer.pth"), weights_only = True))
            res.optim_D.load_state_dict(torch.load(os.path.join(checkpoint_path, "D_optimizer.pth"), weights_only = True))

            if MNE is not None:
                res.MNE = MNE
                res.GE = GE

            return res
        
        # 1. Load from specified checkpoint
        if checkpoint_path is not None:
            return from_checkpoint(root_path, checkpoint_path)

        # 2. Load latest model - with greates number of generator steps
        elif load_latest:
            max_steps, check_label = -1, None
            for label in os.listdir(root_path):
                try:
                    with open(os.path.join(root_path, label, ".metadata.json"), "r", encoding = "utf-8") as f:
                        local_metadata = json.load(f)
                        if local_metadata["g_steps"] > max_steps:
                            max_steps = local_metadata["g_steps"]
                            check_label = label

                except Exception:
                    continue
            
            assert check_label is not None, f"No model checkpoint was foudn in {root_path}"
            return from_checkpoint(root_path, check_label)

        # 3. Load model with best FID on eval - TODO
        elif load_best_fid:
            pass
            
        raise Exception("At least one of checkpoint_path, load_latest, load_best_fid has to be given when loading from trained.")
    
    def __global_metadata_dict(self) -> dict:
        d = {
            "total_steps": self.total_steps,
            "device": self.device,
            "target_resolution": self.target_resolution,
            "batch_size": self.batch_size,
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
            "log_every": self.log_every,
            "mapping_network_params": self.MN.to_dict(),
            "generator_params": self.G.to_dict(),
            "discriminator_params": self.D.to_dict()
        }

        return d

    def __local_metadata_dict(self) -> dict:
        d = {
            "d_steps": self.d_steps,
            "g_steps": self.g_steps,
            "d_loss_ia": float(self.d_loss_ia),
            "g_loss_ia": float(self.g_loss_ia),
            "d_gp_ia": float(self.d_gp_ia),
            "g_plr_ia": float(self.g_plr_ia),
            "save_total_limit": self.save_total_limit,
            "save_history": self.save_history
        }

        return d
    
    def __create_checkpoint(self, check_name : str) -> None:
        # TODO: When FID is implemented, also save model with best FID always, regardless of circumstances
        # Remove oldest checkpoint in save_total_limit regime. save_total_limit is None, this is always False
        if len(self.save_history) == self.save_total_limit:
            target_steps = self.save_history[0]
            
            for label in os.listdir(self.root_path):
                metadata = os.path.join(self.root_path, label, ".metadata.json")
                try:
                    with open(metadata, "r", encoding = "utf-8") as f:
                        metadata = json.load(f)
                    
                    if metadata["g_steps"] == target_steps:
                        shutil.rmtree(os.path.join(self.root_path, label))

                except Exception as e:
                    continue
            
            self.save_history.pop(0)

        self.save_history.append(self.g_steps)
        path = os.path.join(self.root_path, check_name)

        if os.path.exists(path):
            # Re-create checkpoint if it already existed, will delete existing data.
            shutil.rmtree(path)

        os.mkdir(path)

        with open(os.path.join(path, ".metadata.json"), "w+", encoding = "utf-8") as f:
            json.dump(self.__local_metadata_dict(), f, indent = 4)
        
        torch.save(self.MN.state_dict(), os.path.join(path, "MN.pth"))
        torch.save(self.G.state_dict(), os.path.join(path, "G.pth"))
        torch.save(self.D.state_dict(), os.path.join(path, "D.pth"))
        torch.save(self.optim_MN.state_dict(), os.path.join(path, "MN_optimizer.pth"))
        torch.save(self.optim_G.state_dict(), os.path.join(path, "G_optimizer.pth"))
        torch.save(self.optim_D.state_dict(), os.path.join(path, "D_optimizer.pth"))
        
        if self.MNE is not None:
            torch.save(self.MNE.state_dict(), os.path.join(path, "MNE.pth"))
            torch.save(self.GE.state_dict(), os.path.join(path, "GE.pth"))
        
    def __discriminator_step(self, train_loader : torch.utils.data.DataLoader) -> Tuple[float, float]:
        self.optim_D.zero_grad()
        should_reg = self.lazy_reg_steps_discriminator > 0 and (self.d_steps + 1) % self.lazy_reg_steps_discriminator == 0

        d_loss, gp_loss = 0, 0

        for _ in range(self.grad_accum_steps):
            real_samples = next(train_loader).to(self.device)
            # Potential gradient penalty computation
            real_samples.requires_grad_(should_reg)

            real_pred = self.D(real_samples)
            ws, _, _ = new_style_mixing(self.MN, self.batch_size, self.device, self.style_mixing_prob, self.G.num_layers * 2)
            fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device)).detach() # Detach generator from computational graph
            fake_pred = self.D(fake_samples)

            current_d_loss = self.D_loss(real_pred, fake_pred) / self.grad_accum_steps
            current_gp_loss = 0

            d_loss += current_d_loss.item()

            if should_reg:
                # Use R1 gradient penalty for vanilla GAN loss  
                # Quote from StyleGAN2 paper: "We also multiply the regularization term by k to balance the overall magnitude of its gradients"          
                if self.loss_type == "vanilla":
                    current_gp_loss = (self.D_gp(real_samples, real_pred) / self.grad_accum_steps) * self.lazy_reg_steps_discriminator
                    gp_loss += current_gp_loss

                # For WGAN loss we use GP penalty, norm deviations from 1 and fake/real interpolation
                else:
                    ws, _, _ = new_style_mixing(self.MN, self.batch_size, self.device, self.style_mixing_prob, self.G.num_layers * 2)
                    fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
                    current_gp_loss = (self.D_gp(real_samples, real_pred = None, fake_samples = fake_samples, critic = self.D) / self.grad_accum_steps) * self.lazy_reg_steps_discriminator
                    gp_loss += current_gp_loss
            
            current_loss = current_d_loss + current_gp_loss
            current_loss.backward()

        self.optim_D.step()

        self.d_steps += 1
        return d_loss, gp_loss

    # Make private once initial tests are done
    def __generator_step(self) -> Tuple[float, float]:
        self.optim_MN.zero_grad()
        self.optim_G.zero_grad()

        should_reg = self.lazy_reg_steps_generator > 0 and (self.g_steps + 1) % self.lazy_reg_steps_generator == 0

        g_loss, plr_loss = 0, 0

        for _ in range(self.grad_accum_steps):
            ws, _, _ = new_style_mixing(self.MN, self.batch_size, self.device, self.style_mixing_prob, self.G.num_layers * 2)
            fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
            fake_pred = self.D(fake_samples)

            current_g_loss = self.G_loss(fake_pred) / self.grad_accum_steps
            current_plr_loss = 0

            g_loss += current_g_loss

            # Apparently path length regularization uses new fake samples from generator?
            # https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/loss.py#L77
            if should_reg:
                ws, _, _ = new_style_mixing(self.MN, self.batch_size, self.device, self.style_mixing_prob, self.G.num_layers * 2)
                fake_samples = self.G(ws, generate_noise(self.target_resolution, self.batch_size, self.device))
                current_plr_loss = (self.G_plr(ws, fake_samples)  / self.grad_accum_steps) * self.lazy_reg_steps_generator
                plr_loss += current_plr_loss
            
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

        if not (self.compute_generator_ema and self.g_steps >= self.ema_steps_threshold):
            return
        
        if self.GE is None:
            self.GE = copy.deepcopy(self.G).to(self.device).eval()
            self.MNE = copy.deepcopy(self.MN).to(self.device).eval()
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

    def __model_step(self, pbar : tqdm.tqdm, train_loader : torch.utils.data.DataLoader):
        for _ in range(self.disc_optim_steps):
            l, r = self.__discriminator_step(train_loader)
            
            self.d_loss_ia = l if self.d_steps == 1 else ((self.d_steps - 1) / self.d_steps) * self.d_loss_ia + l / self.d_steps

            if self.d_steps % self.lazy_reg_steps_discriminator == 0:
                effective_iter = self.d_steps / self.lazy_reg_steps_discriminator
                self.d_gp_ia = ((effective_iter - 1) / effective_iter) * self.d_gp_ia + r / effective_iter
        
        l, r = self.__generator_step()

        self.g_loss_ia = l if self.g_steps == 1 else ((self.g_steps - 1) / self.g_steps) * self.g_loss_ia + l / self.g_steps

        if self.g_steps % self.lazy_reg_steps_generator == 0:
            effective_iter = self.g_steps / self.lazy_reg_steps_generator
            self.g_plr_ia = ((effective_iter - 1) / effective_iter) * self.g_plr_ia + r / effective_iter

        if self.g_steps % self.log_every == 0:
            pbar.write(f"Steps: {self.g_steps}\nImages generated by generator: {self.g_steps * self.batch_size * self.grad_accum_steps / 1e6:.6f}M\nAverage D loss: {self.d_loss_ia:.4f}\nAverage D gradient penalty: {self.d_gp_ia:.4f}\nAverage G loss: {self.g_loss_ia:.4f}\nAverage G path length regularization: {self.g_plr_ia}\n")

    def train(self, train_loader : torch.utils.data.DataLoader, val_loader : torch.utils.data.DataLoader, num_images_inference : int = 16, num_generated_rows : int = 1):
        pbar = tqdm.tqdm(range(self.g_steps, self.total_steps), position = 0, leave = True)

        for _ in range(self.g_steps, self.total_steps):
            self.__model_step(pbar, train_loader)
            self.__ema_generator_step()
            
            if self.g_steps % self.save_every == 0:
                check_name = f"checkpoint_{self.g_steps}"
                self.__create_checkpoint(check_name)

                # TODO: Sampling only for now, incorporate FID computation
                # Also save which combination of psi and style mixing probability gave best FID score
                with torch.no_grad():
                    for i, psi in enumerate(self.truncation_psi_inference):
                        for j, smp in enumerate(self.style_mixing_prob_inference):
                            # Possible that EMA models don't exist if current g_steps < ema_steps_threshold 
                            if self.MNE is not None:
                                ema_samples = generate_samples(self.GE, self.MNE, self.target_resolution, self.device, num_images_inference, 2 * self.G.num_layers,
                                                                style_mixing_prob = smp,
                                                                truncation_psi = psi,
                                                                num_generated_rows = num_generated_rows,
                                                                update_w_ema = False)
                                    
                                Image.fromarray(ema_samples, mode = "RGB").save(os.path.join(self.root_path, check_name, f"ema_samples_{i}{j}.jpg"))

                            current_samples = generate_samples(self.G, self.MN, self.target_resolution, self.device, num_images_inference, 2 * self.G.num_layers,
                                                            style_mixing_prob = smp,
                                                            truncation_psi = psi,
                                                            num_generated_rows = num_generated_rows,
                                                            update_w_ema = False)
                                
                            Image.fromarray(current_samples, mode = "RGB").save(os.path.join(self.root_path, check_name, f"current_samples_{i}{j}.jpg"))
            pbar.update(1)
        pbar.close()

if __name__ == "__main__":
    DEVICE = "cuda"
    latent_dim = 512
    target_res = 128

    mn = MappingNetwork(latent_dim, 8, lr_mul = 0.01).to(DEVICE)
    g = Generator(target_res, latent_dim).to(DEVICE)
    d = Discriminator(target_res, 3).to(DEVICE)

    train_dataset = Dataset("celeba_128_train")
    eval_dataset = Dataset("celeba_128_eval")

    batch_size = 32
    grad_accum = 4
    train_dl = get_data_loader(train_dataset, batch_size)
    eval_dl = get_data_loader(eval_dataset, batch_size)

    target_num_images = 2e6
    steps = int(target_num_images / (batch_size * grad_accum))
    print(f"Total number of steps: {steps}")
    print(f"Images to train on: {target_num_images}")
    print(f"Number of images per step: {batch_size * grad_accum}")

    t = Trainer(mn, g, d, "stylegan2_celeba_128", steps, DEVICE, target_res, batch_size, 
                loss_type = "vanilla", 
                save_every = 1, 
                learning_rate = 0.002, 
                grad_accum_steps = grad_accum, 
                style_mixing_prob = 0.9, 
                log_every = 10, 
                gen_ema_beta = 0.999, 
                ema_steps_threshold = 3000,
                save_total_limit = 1)
    
    t.train(train_dl, eval_dl, num_generated_rows = 4)