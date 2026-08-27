#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate teacher models saved in GIQA-Guided SSOD checkpoints on a COCO dataset.

This script does not modify training or visualization code. It:
1) builds the detector from a config file,
2) extracts modelTeacher.* weights from each checkpoint,
3) evaluates COCO bbox AP with Detectron2 COCOEvaluator,
4) computes precision/recall/F1 at a chosen score threshold and IoU threshold.

Example:
python tools/eval_teacher.py \
  --config-file configs/plant_giqa_guided_ssod.yaml \
  --weights output_exp1/model_final.pth output_exp2/model_final.pth \
  --coco-json /path/to/test.json \
  --image-root /path/to/test/images \
  --output-csv teacher_eval_results.csv
"""

import argparse
import csv
import json
import os
import sys
from collections import OrderedDict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets.coco import load_coco_json
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.logger import setup_logger

from giqa_ssod import add_giqa_ssod_config
from giqa_ssod.engine.trainer import BaselineTrainer, GIQAGuidedTeacherTrainer

# Register custom modules used by configs.
from giqa_ssod.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN  # noqa: F401
from giqa_ssod.modeling.proposal_generator.rpn import PseudoLabRPN  # noqa: F401
from giqa_ssod.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab  # noqa: F401
import giqa_ssod.data.datasets.builtin  # noqa: F401


CLASS_NAMES = ["plant"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate teacher weights from GIQA-Guided SSOD checkpoints."
    )
    parser.add_argument("--config-file", required=True, help="GIQA-Guided SSOD config YAML.")
    parser.add_argument(
        "--weights",
        nargs="+",
        required=True,
        help="Checkpoint files or directories. If a directory is given, model_final.pth inside it is used.",
    )
    parser.add_argument("--coco-json", required=True, help="COCO annotation json for evaluation.")
    parser.add_argument("--image-root", required=True, help="Image root directory for the COCO json.")
    parser.add_argument("--dataset-name", default="teacher_eval_dataset")
    parser.add_argument("--output-dir", default="teacher_eval_outputs")
    parser.add_argument("--output-csv", default="teacher_eval_results.csv")
    parser.add_argument("--score-thresh", type=float, default=0.5)
    parser.add_argument(
        "--inference-score-thresh",
        type=float,
        default=0.001,
        help="Low model output threshold used during inference. Keep it low for COCO AP; score-thresh is used for precision/recall.",
    )
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument(
        "--teacher-prefix",
        default="modelTeacher.",
        help="Prefix for teacher weights in checkpoints.",
    )
    parser.add_argument(
        "--fallback-prefixes",
        nargs="*",
        default=["modelStudent.", "module.modelTeacher.", "module.modelStudent."],
        help="Fallback prefixes if teacher-prefix is not found.",
    )
    parser.add_argument(
        "--class-aware",
        action="store_true",
        help="Match predictions and GT only when class ids are equal. For single-class plant detection, default false is usually fine.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Dataloader workers for evaluation.",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Additional config options, e.g. MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.05",
    )
    return parser.parse_args()


def resolve_weight_path(path):
    if os.path.isdir(path):
        candidate = os.path.join(path, "model_final.pth")
        if not os.path.exists(candidate):
            raise FileNotFoundError("Directory does not contain model_final.pth: {}".format(path))
        return candidate
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def register_eval_dataset(dataset_name, coco_json, image_root):
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)

    def _loader():
        dataset_dicts = load_coco_json(coco_json, image_root, dataset_name)
        filtered = []
        dropped = 0
        for d in dataset_dicts:
            anns = d.get("annotations", [])
            valid = []
            for ann in anns:
                bbox = ann.get("bbox", [])
                if len(bbox) == 4 and bbox[2] > 0 and bbox[3] > 0:
                    valid.append(ann)
            d["annotations"] = valid
            if valid:
                filtered.append(d)
            else:
                dropped += 1
        print("Loaded {} images, dropped {} images without valid GT boxes.".format(len(filtered), dropped))
        return filtered

    DatasetCatalog.register(dataset_name, _loader)
    MetadataCatalog.get(dataset_name).set(
        thing_classes=CLASS_NAMES,
        evaluator_type="coco",
        json_file=coco_json,
        image_root=image_root,
    )


def setup_cfg(args):
    cfg = get_cfg()
    add_giqa_ssod_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.DATASETS.TEST = (args.dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = float(args.inference_score_thresh)
    cfg.OUTPUT_DIR = args.output_dir
    cfg.freeze()
    return cfg


def safe_checkpoint_name(checkpoint_path):
    parent = os.path.basename(os.path.dirname(os.path.abspath(checkpoint_path)))
    stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
    if parent:
        return "{}__{}".format(parent, stem)
    return stem


def build_model(cfg):
    if cfg.SEMISUPNET.Trainer == "giqa_ssod":
        trainer_cls = GIQAGuidedTeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        trainer_cls = BaselineTrainer
    else:
        raise ValueError("Unknown SEMISUPNET.Trainer: {}".format(cfg.SEMISUPNET.Trainer))
    model = trainer_cls.build_model(cfg)
    model.eval()
    return model


def _strip_prefix_if_present(state_dict, prefix):
    selected = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            selected[key[len(prefix):]] = value
    return selected


def load_teacher_weights(model, checkpoint_path, teacher_prefix, fallback_prefixes):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    clean_state = _strip_prefix_if_present(state_dict, teacher_prefix)
    used_prefix = teacher_prefix

    if not clean_state:
        for prefix in fallback_prefixes:
            clean_state = _strip_prefix_if_present(state_dict, prefix)
            if clean_state:
                used_prefix = prefix
                break

    if not clean_state:
        # Some files may already contain a plain detector state dict.
        clean_state = OrderedDict()
        for key, value in state_dict.items():
            if not key.startswith("modelTeacher.") and not key.startswith("modelStudent."):
                clean_state[key.replace("module.", "", 1)] = value
        used_prefix = "<plain/no-prefix>"

    if not clean_state:
        raise ValueError("No usable model weights found in {}".format(checkpoint_path))

    incompatible = model.load_state_dict(clean_state, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        "Loaded {} using prefix '{}': missing_keys={}, unexpected_keys={}".format(
            checkpoint_path, used_prefix, len(missing), len(unexpected)
        )
    )
    if missing[:5]:
        print("  first missing keys:", missing[:5])
    if unexpected[:5]:
        print("  first unexpected keys:", unexpected[:5])
    return used_prefix


def bbox_xywh_to_xyxy_tensor(annotations, device):
    boxes = []
    classes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        classes.append(int(ann.get("category_id", 0)))
    if boxes:
        return torch.tensor(boxes, dtype=torch.float32, device=device), torch.tensor(
            classes, dtype=torch.long, device=device
        )
    return torch.zeros((0, 4), dtype=torch.float32, device=device), torch.zeros(
        (0,), dtype=torch.long, device=device
    )


def build_gt_lookup(dataset_name):
    gt_lookup = {}
    for d in DatasetCatalog.get(dataset_name):
        gt_lookup[d["image_id"]] = d.get("annotations", [])
    return gt_lookup


def update_pr_counts(outputs, inputs, gt_lookup, counts, score_thresh, iou_thresh, class_aware):
    for output, inp in zip(outputs, inputs):
        instances = output["instances"].to("cpu")
        keep = instances.scores >= score_thresh
        pred_boxes = instances.pred_boxes.tensor[keep]
        pred_scores = instances.scores[keep]
        pred_classes = instances.pred_classes[keep] if instances.has("pred_classes") else None

        annotations = gt_lookup.get(inp.get("image_id"), [])
        gt_boxes_tensor, gt_classes = bbox_xywh_to_xyxy_tensor(annotations, pred_boxes.device)
        num_gt = int(gt_boxes_tensor.shape[0])
        counts["gt"] += num_gt
        counts["pred"] += int(pred_boxes.shape[0])

        if pred_boxes.numel() == 0:
            counts["fn"] += num_gt
            continue
        if num_gt == 0:
            counts["fp"] += int(pred_boxes.shape[0])
            continue

        order = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[order]
        if pred_classes is not None:
            pred_classes = pred_classes[order]

        ious = pairwise_iou(Boxes(pred_boxes), Boxes(gt_boxes_tensor))
        matched_gt = set()
        tp = 0
        fp = 0

        for pred_idx in range(pred_boxes.shape[0]):
            iou_row = ious[pred_idx].clone()
            if class_aware and pred_classes is not None:
                class_mask = gt_classes == pred_classes[pred_idx]
                iou_row[~class_mask] = -1.0
            for gt_idx in matched_gt:
                iou_row[gt_idx] = -1.0
            best_iou, best_gt = torch.max(iou_row, dim=0)
            if float(best_iou) >= iou_thresh:
                tp += 1
                matched_gt.add(int(best_gt))
            else:
                fp += 1

        fn = num_gt - tp
        counts["tp"] += tp
        counts["fp"] += fp
        counts["fn"] += fn


def evaluate_once(model, cfg, dataset_name, checkpoint_path, args):
    run_name = safe_checkpoint_name(checkpoint_path)
    output_folder = os.path.join(args.output_dir, run_name)
    os.makedirs(output_folder, exist_ok=True)

    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = COCOEvaluator(dataset_name, output_dir=output_folder)
    evaluator.reset()

    gt_lookup = build_gt_lookup(dataset_name)
    counts = {"tp": 0, "fp": 0, "fn": 0, "gt": 0, "pred": 0}

    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs)
            evaluator.process(inputs, outputs)
            update_pr_counts(
                outputs,
                inputs,
                gt_lookup,
                counts,
                score_thresh=args.score_thresh,
                iou_thresh=args.iou_thresh,
                class_aware=args.class_aware,
            )

    coco_results = evaluator.evaluate() or {}
    bbox = coco_results.get("bbox", {})

    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    # Detection has no true negatives; this Jaccard-like value is sometimes useful as an accuracy-style summary.
    detection_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    row = {
        "checkpoint": checkpoint_path,
        "AP": bbox.get("AP", float("nan")),
        "AP50": bbox.get("AP50", float("nan")),
        "AP75": bbox.get("AP75", float("nan")),
        "APs": bbox.get("APs", float("nan")),
        "APm": bbox.get("APm", float("nan")),
        "APl": bbox.get("APl", float("nan")),
        "score_thresh": args.score_thresh,
        "iou_thresh": args.iou_thresh,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "num_gt": counts["gt"],
        "num_pred": counts["pred"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "detection_accuracy": detection_accuracy,
    }
    return row, coco_results


def write_csv(rows, output_csv):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    setup_logger()

    weight_paths = [resolve_weight_path(p) for p in args.weights]
    register_eval_dataset(args.dataset_name, args.coco_json, args.image_root)
    cfg = setup_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows = []
    for weight_path in weight_paths:
        print("\n========== Evaluating teacher checkpoint ==========")
        print(weight_path)
        model = build_model(cfg)
        used_prefix = load_teacher_weights(
            model, weight_path, args.teacher_prefix, args.fallback_prefixes
        )
        row, coco_results = evaluate_once(model, cfg, args.dataset_name, weight_path, args)
        row["weight_prefix"] = used_prefix
        all_rows.append(row)

        json_path = os.path.join(
            args.output_dir,
            safe_checkpoint_name(weight_path) + "_metrics.json",
        )
        with open(json_path, "w") as f:
            json.dump({"row": row, "coco": coco_results}, f, indent=2)

        print("Summary:")
        for key in [
            "AP",
            "AP50",
            "AP75",
            "precision",
            "recall",
            "f1",
            "detection_accuracy",
            "tp",
            "fp",
            "fn",
        ]:
            print("  {}: {}".format(key, row[key]))

    write_csv(all_rows, args.output_csv)
    print("\nSaved summary CSV:", os.path.abspath(args.output_csv))


if __name__ == "__main__":
    main()
