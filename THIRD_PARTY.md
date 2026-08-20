# Third-Party Components

This code package contains or interfaces with components derived from publicly available implementations of CLIP, DINOv2, PE-Core, and VisualAD.

The corresponding license and copyright notices are provided in the `LICENSES/` directory.

## CLIP

CLIP-related model-loading and tokenization utilities are included in:

```text
VisualAD_lib/
models/backbone/clip/
```

The corresponding license notice is provided in:

```text
LICENSES/CLIP_LICENSE.txt
```

The pretrained CLIP model weights are not redistributed in this archive.

## VisualAD

Utilities adapted from the publicly available VisualAD implementation are included in:

```text
VisualAD_lib/
```

These utilities support CLIP model construction and local weight loading in the EXPA pipeline.

The corresponding license notice is provided in:

```text
LICENSES/VisualAD_LICENSE.txt
```

## DINOv2

The local DINOv2 model implementation is included in:

```text
models/backbone/dinov2/official/
```

The EXPA-specific feature extraction wrapper is provided in:

```text
models/backbone/dinov2/dinov2_wrapper.py
```

The corresponding license notice is provided in:

```text
LICENSES/DINOv2_LICENSE.txt
```

The pretrained DINOv2 model weights are not redistributed in this archive.

## PE-Core

The PE-Core model implementation and the EXPA-specific feature extraction wrapper are included in:

```text
models/backbone/pe/
utils/pe_wrapper.py
```

The corresponding license notice is provided in:

```text
LICENSES/PE_Core_LICENSE.txt
```

The pretrained PE-Core model weights are not redistributed in this archive.

## EXPA Components

The proposed EXPA components are implemented independently in:

```text
utils/cepa.py
utils/amrf.py
```

These files implement Context-Aware External Prototype Adaptation and Adaptive Multi-level Response Fusion, respectively.
