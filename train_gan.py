import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter
import os
 
# Generator part of the gan
class Generator(nn.Module):

    def __init__(self, nz, nc, ngf):

        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x): return self.net(x)

# Discriminator part of the GAN
class Discriminator(nn.Module):

    def __init__(self, nc, ndf):

        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x): return self.net(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv')!=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# constants and hyperparams
dataroot = "animefacedataset/"
num_epochs = 10
bs = 128
lr = 0.0002
beta1 = 0.5
image_size = 64
nc = 3
nz = 100
ngf, ndf = 64, 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fixed_noise = torch.randn(64, nz, 1, 1, device=device)
real_label, fake_label = 1, 0

# create model, optimizers and loss fn
net_g = Generator(nz, nc, ngf).to(device)
net_g.apply(weights_init)
optimizer_g = torch.optim.Adam(net_g.parameters(), lr=lr, betas=(beta1, 0.999))
print("Generator initialized")
print(net_g)

net_d = Discriminator(nc, ndf).to(device)
net_d.apply(weights_init)
optimizer_d = torch.optim.Adam(net_d.parameters(), lr=lr, betas=(beta1, 0.999))
print("Discriminator initialized")
print(net_d)

criterion = nn.BCELoss()

# create tensorboard writers
writer_real = SummaryWriter(log_dir="logs/real")
writer_fake = SummaryWriter(log_dir="logs/fake")

ds = ImageFolder(
    root = dataroot,
    transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
)    
dl = DataLoader(dataset=ds, batch_size=bs, shuffle=True)

print(f"Number of batches in dataloader: {len(dl)}")

g_losses, d_losses, image_list = [], [], []
iters = 0
print("Starting the training...")
for epoch in range(1, num_epochs+1):
    for i, data in enumerate(dl):

        # optimize the discriminator with both fake and real images
        net_d.zero_grad()
        real_image = data[0].to(device)
        b_size = real_image.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = net_d(real_image).view(-1)
        loss_d_real = criterion(output, label)
        loss_d_real.backward() # backprop the real loss

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = net_g(noise)
        # fill the label with fake label as we want the discriminator to classify them as fake
        # when we are optimising the discriminator
        label.fill_(fake_label)
        # detach the fake images so that the gradient doesn't flow back through the generator 
        output = net_d(fake.detach()).view(-1) 
        loss_d_fake = criterion(output, label)
        loss_d_fake.backward() # backprop the fake loss

        loss_d = loss_d_real + loss_d_fake
        optimizer_d.step() # update the discriminator

        # optimize the generator by making sure it produces images that tricks the discriminator
        # fill the labe with 1 as we want to push the generator to produce realistic images
        net_g.zero_grad()
        label.fill_(real_label)
        output = net_d(fake).view(-1)
        loss_g = criterion(output, label)
        loss_g.backward() # backprop "tricking the discriminator loss"
        optimizer_g.step() # update the generator

        # log results for every 50 steps (simple printing to the terminal for now)
        if i%50==0:
            print(f"[{epoch}/{num_epochs}][{i}/{len(dl)}] loss_d: {loss_d:.4f} loss_g: {loss_g:.4f}")

        g_losses.append(loss_g)
        d_losses.append(loss_d)

        # log real and fake images to tensorboard 
        if iters%500==0 or ((i==len(dl)-1) and (epoch==num_epochs)):
            with torch.inference_mode():
                fake = net_g(fixed_noise).detach().cpu()
            
            real_grid = vutils.make_grid(real_image[:64,:,:,:], padding=2, normalize=True)
            fake_grid = vutils.make_grid(fake, padding=2, normalize=True)
            writer_real.add_image(tag='Real image', img_tensor=real_grid, global_step=iters)
            writer_fake.add_image(tag='Fake image', img_tensor=fake_grid, global_step=iters)

        iters += 1

print("Saving the models...")
if not os.path.exists("checkpoints"):
    os.mkdir("checkpoints")
torch.save(obj={"net_g": net_g.parameters(), "net_d": net_d.parameters()}, f="checkpoints/animegan.pt")
print("Model saved to 'checkpoints/animegan.pt'")