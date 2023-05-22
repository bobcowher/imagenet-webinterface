import torch
import torch.nn as nn
import numpy as np
import cv2
from torchvision.utils import make_grid


device = 'cpu'

root_path = './data'

class Generator(nn.Module):
    def __init__(self, z_dim=64, d_dim=16):
        super(Generator, self).__init__()
        self.z_dim = z_dim

        self.gen = nn.Sequential(
            ## ConvTranspose2d: in_channels, out_channels, kernel_size, stride=1, padding=0
            ## Calculating new width and height: (n-1)*stride - 2*padding +kernel size
            ## Calculating new width and height ()
            ## We begin with a 1x1 image with z_dim number of channels (200)
            nn.ConvTranspose2d(z_dim, d_dim * 32, 4, 1, 0),  ## 4x4 (ch: 200, 512)
            nn.BatchNorm2d(d_dim * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 32, d_dim * 16, 4, 2, 1),  ## 8x8 (ch: 512, 256)
            nn.BatchNorm2d(d_dim * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 16, d_dim * 8, 4, 2, 1),  ## 16x16 (ch: 256, 128)
            nn.BatchNorm2d(d_dim * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 8, d_dim * 4, 4, 2, 1),  ## 32x32 (ch: 128, 64)
            nn.BatchNorm2d(d_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 4, d_dim * 2, 4, 2, 1),  ## 32x32 (ch: 64, 32)
            nn.BatchNorm2d(d_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(d_dim * 2, 3, 4, 2, 1),  ## 32x32 (ch: 32, 3)
            nn.Tanh()
        )

    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)  # 128 x 200 x 1 x 1
        return self.gen(x)


def gen_noise(num, z_dim):
    return torch.randn(num, z_dim)  # 128 x 200


def generate_image():
    checkpoint = torch.load(f"G-latest.pkl", map_location=torch.device('cpu'))
    gen = Generator(200)
    gen.load_state_dict(checkpoint['model_state_dict'])

    noise = gen_noise(1, 200)

    data = gen(noise)

    data = data.detach().cpu()
    grid = make_grid(data[:25], nrow=5).permute(1, 2, 0)
    image = cv2.normalize(grid.numpy(), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    cv2.imwrite(f'./static/images/result.png', image)


def load_checkpoint(name):


    print("Loaded Checkpoint")

