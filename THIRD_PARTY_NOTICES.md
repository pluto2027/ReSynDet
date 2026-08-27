# Third-party notices

This repository interoperates with or derives code from the following
third-party projects:

- [Unbiased Teacher](https://github.com/facebookresearch/unbiased-teacher),
  MIT License. Derived files retain their original copyright headers, and the
  license is included under `GIQA-Guided-SSOD/`.
- [Detectron2](https://github.com/facebookresearch/detectron2), Apache License
  2.0. Detectron2 is installed as an external dependency and is not vendored.
- [PyTorch](https://github.com/pytorch/pytorch) and
  [torchvision](https://github.com/pytorch/vision), used as external
  dependencies under their respective licenses.
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI), GNU GPL v3. The
  generation script communicates with a separately installed ComfyUI server
  through its HTTP API; ComfyUI source code is not included here.

The FLUX.1 Fill, VAE, CLIP, and T5 model files referenced by the supplied
workflow are not distributed in this repository. Users must obtain them from
their official sources and comply with the applicable model licenses.
