import os
import math
import torch
import torch.nn as nn
from copy import deepcopy
from PIL import Image
from tqdm import trange

n_class = 8
n_channel = 64

# Pendefinisian Device GPU dengan CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameter Model DDPM
ema_decay = 0.998
steps = 500
eta = 1.
img_size = 128

seed = 42
class_names = ["Actinic Keratosis", "Basal Cell Carcinoma", "Benign Keratosis",
               "Dermatofibroma", "Melanoma", "Melanocytic Nevi",
               "Squamous Cell", "Vascular Skin Lession"]

# Define the model (a residual U-Net)

class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)

# Fungsi Asli
class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, dropout_last=True):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__([
            nn.Conv2d(c_in, c_mid, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid, c_out, 3, padding=1),
            nn.Dropout2d(0.1, inplace=True) if dropout_last else nn.Identity(),
            nn.ReLU(inplace=True),
        ], skip)

class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        c = n_channel  # The base channel count

        # The inputs to timestep_embed will approximately fall into the range
        # -10 to 10, so use std 0.2 for the Fourier Features.
        self.timestep_embed = FourierFeatures(1, 16, std=0.2)
        self.class_embed = nn.Embedding(n_class, 4)
        self.net = nn.Sequential(   # 128x128
            ResConvBlock(3 + 16 + 4, c, c),
            ResConvBlock(c, c, c),
            SkipBlock([
                nn.AvgPool2d(2),  # 128x128 -> 64x64
                ResConvBlock(c, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c * 2),
                SkipBlock([
                    nn.AvgPool2d(2),  # 64x64 -> 32x32
                    ResConvBlock(c * 2, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 4),
                    SkipBlock([
                        nn.AvgPool2d(2),  # 32x32 -> 16x16
                        ResConvBlock(c * 4, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 8),
                        SkipBlock([
                            nn.AvgPool2d(2),  # 16x16 -> 8x8
                            ResConvBlock(c * 8, c * 16, c * 16),
                            ResConvBlock(c * 16, c * 16, c * 16),
                            SkipBlock([
                                nn.AvgPool2d(2),  # 8x8 -> 4x4
                                ResConvBlock(c * 16, c * 32, c * 32),
                                ResConvBlock(c * 32, c * 32, c * 32),
                                ResConvBlock(c * 32, c * 32, c * 16),
                                nn.Upsample(scale_factor=2),
                            ]),  # 4x4 -> 8x8
                            ResConvBlock(c * 32, c * 16, c * 16),
                            ResConvBlock(c * 16, c * 16, c * 8),
                            nn.Upsample(scale_factor=2),
                        ]),  # 8x8 -> 16x16
                        ResConvBlock(c * 16, c * 8, c * 8),
                        ResConvBlock(c * 8, c * 8, c * 4),
                        nn.Upsample(scale_factor=2),
                    ]),  # 16x16 -> 32x32
                    ResConvBlock(c * 8, c * 4, c * 4),
                    ResConvBlock(c * 4, c * 4, c * 2),
                    nn.Upsample(scale_factor=2),
                ]),  # 32x32 -> 64x64
                ResConvBlock(c * 4, c * 2, c * 2),
                ResConvBlock(c * 2, c * 2, c),
                nn.Upsample(scale_factor=2),
            ]),  # 64x64 -> 128x128
            ResConvBlock(c * 2, c, c),
            ResConvBlock(c, c, 3, dropout_last=False),
        )


    def forward(self, input, log_snrs, cond):
        timestep_embed = expand_to_planes(self.timestep_embed(log_snrs[:, None]), input.shape)
        class_embed = expand_to_planes(self.class_embed(cond), input.shape)
        return self.net(torch.cat([input, class_embed, timestep_embed], dim=1))
    
# Define the noise schedule and sampling loop

def get_alphas_sigmas(log_snrs):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given the log SNR for a timestep."""
    return log_snrs.sigmoid().sqrt(), log_snrs.neg().sigmoid().sqrt()


def get_ddpm_schedule(t):
    """Returns log SNRs for the noise schedule from the DDPM paper."""
    return -torch.special.expm1(1e-4 + 10 * t**2).log()


@torch.no_grad()
def sample(model, x, steps, eta, classes):
    """Draws samples from a model given starting noise."""
    ts = x.new_ones([x.shape[0]])

    # Create the noise schedule
    t = torch.linspace(1, 0, steps + 1)[:-1]
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)

    # The sampling loop
    for i in trange(steps):

        # Get the model output (v, the predicted velocity)
        with torch.amp.autocast('cuda'):
            v = model(x, ts * log_snrs[i], classes).float()

        # Predict the noise and the denoised image
        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v * alphas[i]

        # If we are not on the last timestep, compute the noisy image for the
        # next timestep.
        if i < steps - 1:
            # If eta > 0, adjust the scaling factor for the predicted noise
            # downward according to the amount of additional noise to add
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            # Recombine the predicted noise and predicted denoised image in the
            # correct proportions for the next step
            x = pred * alphas[i + 1] + eps * adjusted_sigma

            # Add the correct amount of fresh noise
            if eta:
                x += torch.randn_like(x) * ddim_sigma
            torch.cuda.empty_cache()  # Tambahkan di sini


    # If we are on the last timestep, output the denoised image
    return pred

checkpoint_path = "weights/ReverseProcessCDDPM.pth"
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
torch.manual_seed(0)
model = Diffusion().to(device)
model_ema = deepcopy(model)

model_ema.load_state_dict(checkpoint['model_ema'])
model_ema.to(device)
model_ema.eval()

@torch.no_grad()
def generate_cddpm(class_input, output_dir: str, count: int):
    """
    Generate multiple synthetic images using Conditional DDPM for the specified class.

    Args:
        class_input (Union[str, int]): Class index (0-7) or class name.
        output_dir (str): Directory to save generated images.
        count (int): Number of images to generate.
    """
    # Validasi kelas
    if isinstance(class_input, int) or (isinstance(class_input, str) and class_input.isdigit()):
        class_index = int(class_input)
        if class_index < 0 or class_index >= len(class_names):
            raise ValueError(f"Class index {class_index} di luar jangkauan.")
    elif isinstance(class_input, str) and class_input in class_names:
        class_index = class_names.index(class_input)
    else:
        raise ValueError(f"Input kelas '{class_input}' tidak valid. Gunakan indeks 0-{len(class_names)-1} atau nama kelas.")

    torch.manual_seed(seed)
    noise = torch.randn([count, 3, img_size, img_size], device=device)
    class_tensor = torch.full((count,), class_index, device=device)

    fakes = sample(model_ema, noise, steps, eta, class_tensor)
    images = (fakes.add(1).div(2).clamp(0, 1) * 255).byte().permute(0, 2, 3, 1).cpu().numpy()

    for i in range(count):
        img = Image.fromarray(images[i])
        img_name = f"{uuid.uuid4()}.png"
        save_path = os.path.join(output_dir, img_name)
        img.save(save_path)
