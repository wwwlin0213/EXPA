# VisualAD Utilities

This directory contains third-party utilities adapted from the publicly available VisualAD implementation and related CLIP utilities.

In the EXPA pipeline, these files are used to support local loading and execution of the CLIP ViT-L/14@336px backbone. In particular, `models/backbone/clip/clip_visualad.py` uses `VisualAD_lib/model_load.py` to construct the CLIP model from a local weight file.

The directory includes model definitions, model-loading utilities, tokenization utilities, and the CLIP byte-pair encoding vocabulary. It does not contain the proposed EXPA components.

The proposed components are implemented separately in:

* `utils/cepa.py`: Context-Aware External Prototype Adaptation;
* `utils/amrf.py`: Adaptive Multi-level Response Fusion.

Pretrained CLIP weights are not included because of the supplementary material submission size limit. The required weight file should be placed at:

```text
models/backbone/clip/weights/ViT-L-14-336px.pt
```

Attribution and license information for VisualAD and CLIP are provided in:

```text
THIRD_PARTY.md
LICENSES/VisualAD_LICENSE.txt
LICENSES/CLIP_LICENSE.txt
```
