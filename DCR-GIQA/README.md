# DCR-GIQA

This directory contains the degradation-chain ranking model used to assign a
quality score to each generated image.

## Method

For every real image, `chain_aug.py` constructs the ordered quality chain
`q = (4, 3, 2, 1, 0)`. `train.py` applies a ResNet-50 scorer and optimizes the
Delta-q-aware pairwise hinge-ranking objective over all ordered pairs:

```text
max(0, alpha * (q_i - q_j) - (s_i - s_j)),  q_i > q_j
```

The ranking objective is the complete training objective. All available real
images are used for the fixed 30-epoch training budget, and
the final checkpoint is saved as `last_model.pth`.

## Training

```bash
python train.py \
  --image-dir /path/to/real/images \
  --image-json /path/to/real_images.json \
  --output-dir outputs/dcr_giqa
```

`--batch-size` counts source-image chains. Each chain contains five image
tensors, one for each quality level.

## Scoring generated images

```bash
python score.py \
  --coco-json /path/to/generated_images.json \
  --image-root /path/to/generated/images \
  --checkpoint outputs/dcr_giqa/last_model.pth \
  --output-csv outputs/dcr_giqa/scores.csv
```

The output columns are `filename`, `raw_score`, and `norm_score`. The normalized
score is min-max scaled to `[0, 100]` over the scored generated-image set.
