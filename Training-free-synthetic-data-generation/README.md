# Training-free synthetic data generation

This directory contains the ComfyUI workflow and batch script used to generate
unlabeled plant images with FLUX.1 Fill. The source image is retained on the
left side of the outpaint canvas, and the newly synthesized right-hand region
is cropped and saved as one generated image.

Start ComfyUI with its API enabled and run:

```bash
python generate.py \
  --input-dir /path/to/real/images \
  --output-dir /path/to/generated/images \
  --server 127.0.0.1:8188 \
  --runs-per-image 20
```

The included workflow requires these ComfyUI model files:

- `flux1-fill-dev.safetensors`
- `ae.safetensors`
- `clip_l.safetensors`
- `t5xxl_fp16.safetensors`

The default positive prompt is `The top-view photograph of leafy greens`.
Only the output of workflow node `9` is downloaded. Existing output files are
skipped unless `--overwrite` is supplied.
