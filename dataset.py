import os
import math
import tqdm
import torch
import torchvision
import torch
import numpy as np

from PIL import Image
from typing import Union, List
from torch.utils.data.distributed import DistributedSampler

def preprocess(img_size, source_dir : str, target_dir : str, allowed_labels : List[str], center_crop_size : Union[float, None] = None, alpha : float = 1):
    """
    CelebA (at least the original form) contains images of size 218x178. 218 = 2 x 89, 178 = 2 x 109, since layers inside
    generator and critic should double/shrink resolutions by factor of 2, this is inconvenient. Hence, we resize all images of CelebA to a 
    desired power of 2. In addition, center cropping can be performed.

    alpha - Controls the portion of the dataset to preprocess, was useful for testing.
    """

    x = int(math.log2(img_size))

    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    clist = [torchvision.transforms.CenterCrop(center_crop_size)] if center_crop_size is not None else []
    compose = torchvision.transforms.Compose(clist + [torchvision.transforms.Resize((img_size, img_size))])

    labels = os.listdir(source_dir)
    labels = labels[:int(len(labels) * alpha)]

    print(f"Processing and transfering CelebA images to {target_dir}. Expected number of images: {len(allowed_labels)}")

    for label in tqdm.tqdm(labels):
        if label in allowed_labels:
            image = Image.open(os.path.join(source_dir, label))
            image = compose(image)
            image.save(os.path.join(target_dir, label))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path : str, alpha : float = 1):
        """
        Constructor for custom RGB image Dataset class. 
        path - Name of the directory that contains images, preferably resized to a power of 2. Path should be given relative to cwd.
        alpha - Ratio of original dataset to consider. Was useful for testing
        """
        self.path = path
        self.alpha = alpha
        self.imgs = [os.path.join(path, item) for item in os.listdir(self.path)]
        self.imgs = self.imgs[:int(len(self.imgs) * alpha)]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) -> torch.Tensor:
        return torch.tensor(np.asarray(Image.open(self.imgs[idx])).transpose((2, 0, 1)).astype(np.float32))
        
def get_data_loader(dataset : Dataset, batch_size : int, is_ddp : bool, pin_memory : bool = True, num_workers : int = 0):
    if is_ddp:
        dl = torch.utils.data.DataLoader(
                                       dataset, 
                                       batch_size = batch_size, 
                                       shuffle = False, 
                                       collate_fn = lambda x: torch.stack(x, dim = 0) / 127.5 - 1, # Transform to range [-1, 1]
                                       drop_last = True,
                                       pin_memory = pin_memory,
                                       num_workers = num_workers,
                                       sampler = DistributedSampler(dataset, shuffle = True)) 
        
    else:
        dl = torch.utils.data.DataLoader(
                                       dataset, 
                                       batch_size = batch_size, 
                                       shuffle = True, 
                                       pin_memory = pin_memory, 
                                       num_workers = num_workers, 
                                       collate_fn = lambda x: torch.stack(x, dim = 0) / 127.5 - 1, # Transform to range [-1, 1]
                                       drop_last = True) 
 
    while True:
        for sample in dl:
            yield sample