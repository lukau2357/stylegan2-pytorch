import argparse
import torch
import os
import time
import json

from datetime import datetime

from model import MappingNetwork, Generator
from utils import generate_samples
from PIL import Image

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("model_dir", type = str, help = "Model directory path, which contains .pth files for the trained StyleGAN2 instance.")
    args.add_argument("inference_dir", type = str, help = "Directory in which inferred images should be saved.")
    args.add_argument("--truncation_psi", type = float, default = 0.6, help = "Psi parameter for truncation trick, must be in [0, 1]. 1 Means no truncation, 0 means all samples coincide with estimated mean style vector.")
    args.add_argument("--inference_count", type = int, default = 1, help = "Determines the number of times the inference will be repeated.")
    args.add_argument("--num_samples", type = int, default = 16, help = "Number of samples to generate.")
    args.add_argument("--num_rows", type = int, default = 4, help = "How many rows should the output sample grid have. Shoud evenly divide --num_samples.")
    args.add_argument("--random_seed", type = int, default = 1337, help = "Seed used for generation.")
    args.add_argument("--no_ema", type = bool, default = True, const = False, nargs = "?", help = "Use EMA of generator weights or use generator from last training iteration. Defaults to True.")
    args.add_argument("--w_estimate_samples", type = int, default = 20000, help = "If mapping network did not record EMA of style vectors seen during training, this parameter determines the number of samples used to estimate the mean style vector.")
    return args.parse_args()

def inference(model_dir : str, inference_dir : str, device : str, truncation_psi : float, inference_count : int, num_samples : int , num_rows : int, use_ema : bool, w_estimate_samples : int):
    if use_ema:
        mn_d = torch.load(os.path.join(model_dir, "MNE.pth"), weights_only = True)
        g_d = torch.load(os.path.join(model_dir, "GE.pth"), weights_only = True)
    
    else:
        mn_d = torch.load(os.path.join(model_dir, "MN.pth"), weights_only = True)
        g_d = torch.load(os.path.join(model_dir, "G.pth"), weights_only = True)

    mapping_network = MappingNetwork.from_dict(mn_d).to(device)
    generator = Generator.from_dict(g_d).to(device)

    print(f"Found model in {model_dir}. Generator successfully loaded.")
    times = []

    timestamp = str(int(time.time()))
    os.mkdir(os.path.join(inference_dir, timestamp))

    metadata = {
        "truncation_psi": truncation_psi,
        "inference_count": inference_count,
        "use_ema": use_ema,
        "w_estimate_samples": w_estimate_samples,
        "mapping_network_metadata": mapping_network.to_dict(),
        "generator_params_metadata": generator.to_dict()
    }

    with open(os.path.join(inference_dir, timestamp, ".run_metadata.json"), "w+", encoding = "utf-8") as f:
        json.dump(metadata, f, indent = 4)

    for i in range(inference_count):
        start = time.time()
        grid, truncation_base = generate_samples(generator, mapping_network, device, num_samples, 
                                style_mixing_prob = 0, # No style mixing for inference
                                truncation_psi = truncation_psi,
                                num_generated_rows = num_rows,
                                w_estimate_samples = w_estimate_samples,
                                compute_truncation_base = True)
        end = time.time()
        times.append(end - start)
        Image.fromarray(grid, mode = "RGB").save(f"{os.path.join(inference_dir, timestamp, str(i))}.png")
        Image.fromarray(truncation_base, mode = "RGB").save(f"{os.path.join(inference_dir, timestamp, f'truncation_base_{i}')}.png")

    print(f"Mean inference time: {sum(times) / inference_count}s.")
    
if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.inference_dir):
        os.mkdir(args.inference_dir)
    
    datetime_str = datetime.today().strftime(r"%Y_%m_%d")
    inference_dir = os.path.join(args.inference_dir, datetime_str)

    if not os.path.exists(inference_dir):
        os.mkdir(inference_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu" # Single GPU mode for inference
    torch.manual_seed(args.random_seed)
    inference(args.model_dir, inference_dir, device, args.truncation_psi, args.inference_count, args.num_samples, args.num_rows, args.no_ema, args.w_estimate_samples)