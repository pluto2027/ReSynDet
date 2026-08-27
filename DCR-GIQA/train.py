"""Train DCR-GIQA with an ordered degradation-chain ranking objective."""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chain_aug import QUALITY_LEVELS, apply_quality_aug


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(_worker_id):
    random.seed(torch.initial_seed() % (2**32))


class DegradationChainDataset(Dataset):
    def __init__(self, image_paths, quality_levels=QUALITY_LEVELS):
        self.image_paths = list(image_paths)
        self.quality_levels = tuple(quality_levels)
        if any(
            first <= second
            for first, second in zip(self.quality_levels, self.quality_levels[1:])
        ):
            raise ValueError("quality_levels must be strictly descending.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with Image.open(self.image_paths[index]) as image:
            image = image.convert("RGB")
            chain = [apply_quality_aug(image, q) for q in self.quality_levels]
        quality = torch.tensor(self.quality_levels, dtype=torch.float32)
        return torch.stack(chain), quality


def build_resnet50(pretrained):
    try:
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        return models.resnet50(weights=weights)
    except AttributeError:  # torchvision < 0.13
        return models.resnet50(pretrained=pretrained)


class DCRGIQAScorer(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = build_resnet50(pretrained)
        feature_dim = backbone.fc.in_features
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.score_head = nn.Linear(feature_dim, 1)

    def forward(self, images):
        features = self.features(images).flatten(1)
        return self.score_head(features).squeeze(1)


def degradation_ranking_loss(scores, quality, alpha=0.5):
    """Delta-q-aware hinge ranking loss over all pairs in each chain."""
    if scores.shape != quality.shape or scores.ndim != 2:
        raise ValueError("scores and quality must have the same (batch, chain) shape.")

    chain_length = scores.shape[1]
    idx_high, idx_low = torch.triu_indices(
        chain_length, chain_length, offset=1, device=scores.device
    )
    quality_gap = quality[:, idx_high] - quality[:, idx_low]
    if torch.any(quality_gap <= 0):
        raise ValueError("Quality levels must be strictly descending within each chain.")

    score_gap = scores[:, idx_high] - scores[:, idx_low]
    return torch.clamp(alpha * quality_gap - score_gap, min=0.0).mean()


def collect_image_paths(image_dir, image_json):
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(image_dir)

    if image_json:
        with open(image_json, "r", encoding="utf-8") as handle:
            coco = json.load(handle)
        paths = []
        for record in coco.get("images", []):
            file_name = record.get("file_name", "")
            path = Path(file_name)
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path if path.is_absolute() else image_dir / path)
    else:
        paths = [
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    paths = sorted(path.resolve() for path in paths)
    if not paths:
        raise ValueError("No training images were found.")
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate image paths were found in the training set.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing training images:\n  " + "\n  ".join(missing[:10]))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument(
        "--image-json",
        default="",
        help="Optional COCO JSON selecting images relative to --image-dir.",
    )
    parser.add_argument("--output-dir", default="outputs/dcr_giqa")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.5)
    return parser.parse_args()


def train(args):
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.alpha <= 0:
        raise ValueError("--alpha must be positive.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_paths = collect_image_paths(args.image_dir, args.image_json)
    dataset = DegradationChainDataset(image_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )

    model = DCRGIQAScorer(pretrained=True).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print("Using {} visible GPUs with DataParallel.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    print("Device: {}".format(device))
    print("Training images: {}".format(len(dataset)))
    print(
        "Chains per batch: {}; images per full batch: {}".format(
            args.batch_size, args.batch_size * len(QUALITY_LEVELS)
        )
    )

    final_loss = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for image_chain, quality_chain in loader:
            batch_size, chain_length, channels, height, width = image_chain.shape
            images = image_chain.reshape(
                batch_size * chain_length, channels, height, width
            ).to(device, non_blocking=True)
            quality_chain = quality_chain.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            scores = model(images).reshape(batch_size, chain_length)
            loss = degradation_ranking_loss(scores, quality_chain, alpha=args.alpha)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach()) * batch_size
            sample_count += batch_size

        final_loss = loss_sum / sample_count
        print(
            "[Epoch {:03d}/{:03d}] ranking_loss={:.6f}".format(
                epoch, args.epochs, final_loss
            )
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "last_model.pth"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ranking_loss": final_loss,
            "quality_levels": QUALITY_LEVELS,
            "alpha": args.alpha,
            "num_real_images": len(dataset),
            "seed": args.seed,
        },
        checkpoint_path,
    )
    print("Saved final checkpoint: {}".format(checkpoint_path))


if __name__ == "__main__":
    train(parse_args())
