"""Synthetic degradation chain used to train DCR-GIQA.

Each source image is transformed into five ordered quality levels. Geometry is
fixed (resize followed by center crop), so the ranking task cannot be solved by
comparing different image regions.
"""

import random

import torch
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter


QUALITY_LEVELS = (4, 3, 2, 1, 0)

base_preprocess = T.Compose([T.Resize(256), T.CenterCrop(224)])
to_pil = T.ToPILImage()
to_tensor = T.ToTensor()
_RESAMPLING = getattr(Image, "Resampling", Image)


def add_random_color_artifacts(image, num_blobs_range=(3, 8)):
    """Add spatially localized color artifacts to an RGB tensor."""
    channels, height, width = image.shape
    if channels != 3:
        raise ValueError("Color artifacts require a three-channel RGB tensor.")

    device = image.device
    try:
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
            indexing="ij",
        )
    except TypeError:  # torch < 1.10
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device),
            torch.arange(width, device=device),
        )

    scale_map = torch.ones_like(image)
    for _ in range(random.randint(*num_blobs_range)):
        center_y = random.randint(0, height - 1)
        center_x = random.randint(0, width - 1)
        radius = random.randint(max(1, height // 20), max(1, height // 6))
        distance_squared = (yy - center_y) ** 2 + (xx - center_x) ** 2
        mask = (distance_squared <= radius * radius).to(image.dtype)
        scales = torch.tensor(
            [
                1.0 + random.uniform(0.3, 0.6),
                1.0 - random.uniform(0.0, 0.6),
                1.0 + random.uniform(-0.6, 0.6),
            ],
            dtype=image.dtype,
            device=device,
        ).view(3, 1, 1)
        scale_map = scale_map * (1.0 - mask) + scale_map * mask * scales

    return torch.clamp(image * scale_map, 0.0, 1.0)


def add_gaussian_noise(image, sigma_range):
    sigma = random.uniform(*sigma_range)
    return torch.clamp(image + torch.randn_like(image) * sigma, 0.0, 1.0)


def down_up_sample(image, scale_range):
    scale = random.uniform(*scale_range)
    width, height = image.size
    reduced_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    reduced = image.resize(reduced_size, resample=_RESAMPLING.BILINEAR)
    return reduced.resize((width, height), resample=_RESAMPLING.BILINEAR)


def degrade_color(image, strength_range):
    strength = random.uniform(*strength_range)
    image = ImageEnhance.Contrast(image).enhance(1.0 - 0.4 * strength)
    image = ImageEnhance.Color(image).enhance(1.0 - 0.6 * strength)
    image = ImageEnhance.Brightness(image).enhance(1.0 - 0.2 * strength)
    return T.ColorJitter(hue=0.08 * strength)(image)


def repeat_random_patch(image, min_scale=0.15, max_scale=0.35):
    """Tile a randomly selected patch to simulate severe structural artifacts."""
    width, height = image.size
    scale = random.uniform(min_scale, max_scale)
    patch_width = max(1, int(width * scale))
    patch_height = max(1, int(height * scale))
    left = random.randint(0, max(0, width - patch_width))
    top = random.randint(0, max(0, height - patch_height))
    patch = image.crop((left, top, left + patch_width, top + patch_height))

    output = Image.new("RGB", (width, height))
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            output.paste(patch, (x, y))
    return output


def apply_quality_aug(image, quality):
    """Return one member of the ordered degradation chain as a tensor."""
    if quality not in QUALITY_LEVELS:
        raise ValueError(
            "quality must be one of {}; received {}".format(QUALITY_LEVELS, quality)
        )

    image = base_preprocess(image.convert("RGB"))
    if quality == 4:
        return to_tensor(image)

    parameters = {
        3: {"blur": (0.4, 0.5), "color": (0.2, 0.3), "scale": None, "noise": None},
        2: {"blur": (1.0, 1.1), "color": (0.4, 0.5), "scale": (0.80, 0.85), "noise": (0.01, 0.02)},
        1: {"blur": (1.6, 1.7), "color": (0.7, 0.8), "scale": (0.60, 0.65), "noise": (0.04, 0.05)},
        0: {"blur": (2.2, 3.0), "color": (0.9, 1.1), "scale": (0.30, 0.35), "noise": (0.08, 0.10)},
    }
    params = parameters[quality]

    image = image.filter(
        ImageFilter.GaussianBlur(radius=random.uniform(*params["blur"]))
    )
    if params["scale"] is not None:
        image = down_up_sample(image, params["scale"])
    image = degrade_color(image, params["color"])

    tensor = to_tensor(image)
    if params["noise"] is not None:
        tensor = add_gaussian_noise(tensor, params["noise"])

    if quality == 0:
        image = to_pil(tensor).filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.0))
        )
        if random.random() < 0.4:
            image = repeat_random_patch(image)
        tensor = to_tensor(image)
        if random.random() < 0.4:
            tensor = add_random_color_artifacts(tensor)

    return tensor
