import torch
from torchvision import utils as vutils
from train_gan import Generator, nz, nc, ngf
import numpy as np
import random
import streamlit as st
from PIL import Image
from io import BytesIO

# Initialize model and params
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net_g = Generator(nz, nc, ngf).to(device)
net_g.load_state_dict(torch.load("checkpoints/animegan.pt", map_location=device)['net_g'])

st.title("AnimeGAN")
st.write("Generate anime faces")

# Add batch size and seed input via sliders
bs = st.slider("Batch Size", min_value=1, max_value=64, value=64)
seed = st.slider("Seed", min_value=1, max_value=32000, value=None)

# Generate Image button
if st.button("Generate Image"):
    # Set manual seed
    if seed: torch.random.manual_seed(seed)
    else:
        seed = random.randint(1, 32000)
        torch.random.manual_seed(seed)
    st.write(f"Seed set to {seed}")
    # Generate noise and create images
    noise = torch.randn(bs, nz, 1, 1, device=device)
    fake_images = net_g(noise)
    # Create a grid of images and convert to a format for display and download
    grid = vutils.make_grid(fake_images.to(device), padding=2, normalize=True).cpu()
    grid_np = np.transpose(grid.numpy(), (1, 2, 0))
    # Display the image grid
    st.image(grid_np, use_column_width=True)
    # Convert the image grid to a PIL image for downloading
    img = Image.fromarray((grid_np * 255).astype(np.uint8))
    # Save the image to a buffer
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    # Provide a download button
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="generated_image.jpg",
        mime="image/jpeg"
    )