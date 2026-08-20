# DINOv2 Weights

The pretrained DINOv2 weights are not included in this archive because they exceed the supplementary material submission size limit.

Please obtain the pretrained DINOv2 ViT-L/14 weights from the official source and place the file at:

```text
models/backbone/dinov2/weights/dinov2_vitl14_pretrain.pth
```

The expected directory structure is:

```text
models/backbone/dinov2/weights/
├── README.md
└── dinov2_vitl14_pretrain.pth
```

This weight file is used when the backbone is set to:

```text
dinov2_vitl14
```
