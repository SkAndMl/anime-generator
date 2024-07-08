import argparse
import os
import torch
from torchvision import utils as vutils
from train_gan import Generator, nz, nc, ngf
import matplotlib.pyplot as plt
import numpy as np
import random

# initialize model and params
print(f"Initializing generator")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_g = Generator(nz, nc, ngf).to(device)
net_g.load_state_dict(torch.load("checkpoints/animegan.pt", map_location=device)['net_g'])

print("Initialized model. Ready for inference...")

# add batch size and image saving location to the cli
parser = argparse.ArgumentParser()
parser.add_argument("--bs", type=int, default=64)
parser.add_argument("--img_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()
bs = args.bs
img_path = args.img_path
seed = args.seed

# create folder if it does not exist
os.makedirs(os.path.dirname(img_path), exist_ok=True)

if seed: torch.random.manual_seed(seed)
else:
    print(f"Manual seed not passed. Setting own seed...")
    seed = random.randint(-2147483648, 2147483647)
    torch.random.manual_seed(seed)
print(f"Seed set to {seed}")


# create noise and pass it through the gen
noise = torch.randn(bs, nz, 1, 1, device=device)
fake_images = net_g(noise)

# put images in a grid and save it to img_path
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.imshow(np.transpose(vutils.make_grid(fake_images.to(device), padding=2, normalize=True).cpu(),(1,2,0)))
plt.savefig(img_path)

print(f"Images have been generated and saved at {img_path}")