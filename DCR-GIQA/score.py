"""Score generated images with a trained DCR-GIQA model."""

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from chain_aug import base_preprocess, to_tensor
from train import DCRGIQAScorer


class CocoImageDataset(Dataset):
    def __init__(self, coco_json, image_root):
        with open(coco_json, "r", encoding="utf-8") as handle:
            coco = json.load(handle)

        image_root = Path(image_root).expanduser().resolve()
        self.items = []
        seen_names = set()
        for record in coco.get("images", []):
            file_name = record.get("file_name", "")
            path = Path(file_name)
            path = path if path.is_absolute() else image_root / path
            basename = Path(file_name).name
            if not basename:
                raise ValueError("An image record has an empty file_name.")
            if basename in seen_names:
                raise ValueError("Duplicate image basename: {}".format(basename))
            if not path.is_file():
                raise FileNotFoundError(path)
            seen_names.add(basename)
            self.items.append((basename, path))

        if not self.items:
            raise ValueError("The COCO JSON contains no images.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        basename, path = self.items[index]
        with Image.open(path) as image:
            tensor = to_tensor(base_preprocess(image.convert("RGB")))
        return basename, tensor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def remove_module_prefix(state_dict):
    return {
        key[7:] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = CocoImageDataset(args.coco_json, args.image_root)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint.get("model_state", checkpoint)
    model = DCRGIQAScorer(pretrained=False)
    model.load_state_dict(remove_module_prefix(state_dict), strict=True)
    model.to(device).eval()

    scored = []
    with torch.no_grad():
        for basenames, images in loader:
            scores = model(images.to(device, non_blocking=True)).cpu().tolist()
            scored.extend(zip(basenames, scores))

    raw_scores = [score for _, score in scored]
    score_min = min(raw_scores)
    score_max = max(raw_scores)
    if score_max == score_min:
        raise ValueError("All raw scores are identical; min-max normalization is undefined.")

    rows = [
        {
            "filename": filename,
            "raw_score": "{:.8f}".format(raw_score),
            "norm_score": "{:.6f}".format(
                (raw_score - score_min) / (score_max - score_min) * 100.0
            ),
        }
        for filename, raw_score in scored
    ]
    rows.sort(key=lambda row: row["filename"])

    output_path = Path(args.output_csv).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("filename", "raw_score", "norm_score")
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Scored {} images.".format(len(rows)))
    print("Raw score range: [{:.6f}, {:.6f}]".format(score_min, score_max))
    print("Saved: {}".format(output_path))


if __name__ == "__main__":
    main()
