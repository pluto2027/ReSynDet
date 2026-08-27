# GIQA-Guided SSOD

This directory contains the semi-supervised detector used with DCR-GIQA
scores. It is implemented on top of
[Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher). The
derived Python package is named `giqa_ssod`; original copyright notices and
the upstream MIT License are retained.

## What is changed

For each unlabeled image, `giqa_ssod.data.dataset_mapper` reads its DCR-GIQA
score and maps it linearly to an image weight. For normalized score `s`, the
raw weight is `clip(w_min + (w_max - w_min) * s, w_min, w_max)`. When mean
normalization is enabled, all raw weights are rescaled to the configured target
mean. The resulting weight is propagated through the detector and applied to
the unlabeled RPN and ROI losses. Labeled images always receive weight 1.
Pseudo-label generation by the teacher is unchanged.

The three supplied configs share the same code and training settings:

- `plant_baseline.yaml`: `UNSUP_LOSS_WEIGHT=0`.
- `plant_naive_ssod.yaml`: standard unweighted SSOD.
- `plant_giqa_guided_ssod.yaml`: GIQA-weighted SSOD.

## Data format

Labeled, unlabeled, and validation sets use COCO JSON. The unlabeled JSON needs
only the `images` and `categories` fields. GIQA scores use the columns shown in
`examples/scores_example.csv`; score matching uses the image basename.

Every unlabeled image should have exactly one score row. A missing score is
assigned raw weight 1 before optional dataset-level mean normalization.

## Training

Create the tested software environment with `environment.yml`, then run:

```bash
python train_net.py \
  --num-gpus 4 \
  --config-file configs/plant_giqa_guided_ssod.yaml \
  --labeled-json /path/to/labeled.json \
  --labeled-root /path/to/labeled/images \
  --unlabeled-json /path/to/unlabeled.json \
  --unlabeled-root /path/to/generated/images \
  --val-json /path/to/validation.json \
  --val-root /path/to/validation/images \
  SEMISUPNET.IQA_SCORE_PATH /path/to/scores.csv
```

Use `plant_baseline.yaml` or `plant_naive_ssod.yaml` for the corresponding
comparison without changing the training code.

Without `--resume`, `MODEL.WEIGHTS` is loaded as a plain detector checkpoint
into the student and then copied to the teacher, matching the original plant
experiments. With `--resume`, the input is treated as a complete
teacher-student training checkpoint including optimizer and iteration state.

## Teacher evaluation

The evaluation utility extracts the teacher state from each checkpoint and
uses Detectron2's `COCOEvaluator` for bbox AP:

```bash
python tools/eval_teacher.py \
  --config-file configs/plant_giqa_guided_ssod.yaml \
  --weights /path/to/model_final.pth \
  --coco-json /path/to/test.json \
  --image-root /path/to/test/images \
  --score-thresh 0.6 \
  --inference-score-thresh 0.001 \
  --iou-thresh 0.75
```
