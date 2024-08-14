import torch
import torchvision
import os
import cv2
import numpy as np

from model import MappingNetwork

if __name__ == "__main__":
    pth = os.path.join("img_align_celeba", "img_align_celeba", "000001.jpg")
    img = cv2.imread(pth)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img)
    img = torchvision.transforms.Resize((128, 128))(np.transpose(img, (2, 0, 1)))
    
    grid = torchvision.utils.make_grid([img], nrow = 1)
    grid = torchvision.transforms.ToPILImage()(grid)
    grid.show()
