# EXPA Code Supplement

This archive contains the training and evaluation code for EXPA, an external prototype adaptation framework for zero-shot visual anomaly detection.

The main cross-dataset experiments train on one industrial anomaly detection dataset and directly evaluate on another dataset without target-domain training, fine-tuning, test-time adaptation, or target-domain reference images.

The default configuration uses the PE-Core-L14-336 visual backbone. CLIP ViT-L/14@336px and DINOv2 ViT-L/14 are also supported for backbone comparisons.

## Directory Structure

```text
.
├── configs/
│   └── backbone_layers.yaml
├── generate_dataset_json/
├── LICENSES/
├── metadata/
│   ├── mvtec.json
│   ├── visa.json
│   └── README.md
├── models/
│   └── backbone/
│       ├── clip/
│       ├── dinov2/
│       └── pe/
├── scripts/
│   └── run_cross_dataset.sh
├── utils/
│   ├── cepa.py
│   ├── amrf.py
│   └── ...
├── VisualAD_lib/
├── dataset.py
├── train.py
├── test.py
├── requirements.txt
├── EXPECTED_RESULTS.md
└── THIRD_PARTY.md
```

The proposed EXPA components are implemented in:

```text
utils/cepa.py
utils/amrf.py
```

## Environment

The code was tested with:

```text
Python 3.10
PyTorch 2.11.0
CUDA-enabled GPU
```

Install PyTorch and torchvision separately using versions compatible with the local CUDA environment. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The CEPA implementation uses a CUDA extension. A compatible CUDA toolkit, including `nvcc`, and the `ninja` build tool are therefore required.

All commands should be executed from the project root directory:

```bash
cd EXPA-Code-Submission
```

## Pretrained Backbone Weights

Pretrained backbone weights are not included because of the supplementary material submission size limit.

Place the required weights at the following locations.

### PE-Core-L14-336

```text
models/backbone/pe/weights/PE-Core-L14-336.pt
```

Backbone identifier:

```text
pe_l14_336
```

### CLIP ViT-L/14@336px

```text
models/backbone/clip/weights/ViT-L-14-336px.pt
```

Backbone identifier:

```text
clip_vitl14_336
```

### DINOv2 ViT-L/14

```text
models/backbone/dinov2/weights/dinov2_vitl14_pretrain.pth
```

Backbone identifier:

```text
dinov2_vitl14
```

Additional instructions are provided in the `README.md` file under each weights directory.

## Dataset Preparation

Download and extract MVTec AD and VisA using their standard directory structures.

The provided metadata files are:

```text
metadata/mvtec.json
metadata/visa.json
```

These JSON files contain relative image and mask paths. Therefore, the paths supplied to `train.py` and `test.py` must point to the corresponding dataset root directories.

Example dataset locations:

```text
/path/to/MVTec_AD
/path/to/VisA
```

The original dataset images are not included in this archive.

The metadata files used for the reported MVTec AD and VisA experiments are already provided. Regenerating them is not required. Reference preprocessing scripts for other datasets are available under:

```text
generate_dataset_json/
```

## Main Cross-Dataset Experiments

The provided script runs both main transfer directions:

1. MVTec AD to VisA;
2. VisA to MVTec AD.

Before execution, edit the following variables in `scripts/run_cross_dataset.sh`:

```bash
MVTEC_PATH="/path/to/MVTec_AD"
VISA_PATH="/path/to/VisA"
GPU="cuda:0"
```

The default experiment configuration is:

```text
Backbone: PE-Core-L14-336
Input resolution: 518 × 518
Feature layers: 6, 12, 18, and 24
Batch size: 8
CEPA: enabled
AMRF: enabled
MVTec AD to VisA: 2 epochs
VisA to MVTec AD: 1 epoch
```

Run both experiments with:

```bash
bash scripts/run_cross_dataset.sh
```

The script performs training followed by evaluation for each transfer direction.

## Manual Training

### MVTec AD to VisA

```bash
python train.py \
    --train_data_path /path/to/MVTec_AD \
    --save_path experiments/pe_l14_336/mvtec_to_visa/checkpoints \
    --train_dataset mvtec \
    --backbone pe_l14_336 \
    --features_list 6 12 18 24 \
    --use_cepa true \
    --use_amrf true \
    --epoch 2 \
    --batch_size 8 \
    --image_size 518 \
    --seed 111 \
    --device cuda:0
```

### VisA to MVTec AD

```bash
python train.py \
    --train_data_path /path/to/VisA \
    --save_path experiments/pe_l14_336/visa_to_mvtec/checkpoints \
    --train_dataset visa \
    --backbone pe_l14_336 \
    --features_list 6 12 18 24 \
    --use_cepa true \
    --use_amrf true \
    --epoch 1 \
    --batch_size 8 \
    --image_size 518 \
    --seed 111 \
    --device cuda:0
```

## Manual Evaluation

### Evaluate on VisA

```bash
python test.py \
    --test_data_path /path/to/VisA \
    --checkpoint_path experiments/pe_l14_336/mvtec_to_visa/checkpoints/epoch_2.pth \
    --test_dataset visa \
    --save_path experiments/pe_l14_336/mvtec_to_visa/results/epoch_2 \
    --sigma 4 \
    --device cuda:0 \
    --use_cepa true \
    --use_amrf true
```

### Evaluate on MVTec AD

```bash
python test.py \
    --test_data_path /path/to/MVTec_AD \
    --checkpoint_path experiments/pe_l14_336/visa_to_mvtec/checkpoints/epoch_1.pth \
    --test_dataset mvtec \
    --save_path experiments/pe_l14_336/visa_to_mvtec/results/epoch_1 \
    --sigma 4 \
    --device cuda:0 \
    --use_cepa true \
    --use_amrf true
```

The backbone, input resolution, feature layers, and module configurations are restored from the selected checkpoint during evaluation.

Use the `--enable_analysis` flag to additionally generate class-wise analysis and visualization outputs:

```bash
python test.py \
    ... \
    --enable_analysis
```

## Outputs

With the default script, checkpoints and evaluation results are saved under:

```text
experiments/
└── pe_l14_336/
    ├── mvtec_to_visa/
    │   ├── checkpoints/
    │   │   └── epoch_2.pth
    │   └── results/
    │       └── epoch_2/
    └── visa_to_mvtec/
        ├── checkpoints/
        │   └── epoch_1.pth
        └── results/
            └── epoch_1/
```

Execution logs are saved under:

```text
logs/pe_l14_336/
```

Trained checkpoints are not included in this archive because they exceed the submission size limit.

## Expected Results

Representative results for the default PE-Core-L14-336 configuration are reported in:

```text
EXPECTED_RESULTS.md
```

The expected cross-dataset results are:

| Transfer direction | I-AUC | P-AUC |  PRO |
| ------------------ | ----: | ----: | ---: |
| MVTec AD to VisA   |  90.4 |  95.7 | 93.0 |
| VisA to MVTec AD   |  93.7 |  90.9 | 88.1 |

Minor numerical differences may arise from the CUDA version, GPU architecture, and nondeterministic CUDA operators.

## Third-Party Components

The package contains or interfaces with publicly available implementations of PE-Core, CLIP, DINOv2, and VisualAD-related utilities.

Attribution and license information are provided in:

```text
THIRD_PARTY.md
LICENSES/
```
