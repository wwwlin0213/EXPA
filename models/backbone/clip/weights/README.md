# CLIP Weights

The pretrained CLIP weights are not included in this archive because they exceed the supplementary material submission size limit.

Please obtain the pretrained CLIP ViT-L/14@336px weights from the official source and place the file at:

```text
models/backbone/clip/weights/ViT-L-14-336px.pt
```

The expected directory structure is:

```text
models/backbone/clip/weights/
├── README.md
└── ViT-L-14-336px.pt
```

This weight file is used when the backbone is set to:

```text
clip_vitl14_336
```
