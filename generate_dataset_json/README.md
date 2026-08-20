# Dataset Metadata Generation

This directory contains preprocessing scripts for generating dataset metadata in the JSON format expected by `dataset.py`.

The scripts cover the industrial and medical datasets used in the experiments, including:

```text
DAGM
DTD-Synthetic
KSDD
BTAD
MVTec AD
VisA
OCT2017
Brain AD
Br35H
EndoTect 2020
Kvasir
```

## Provided Metadata

The exact metadata files used for the main cross-dataset experiments are already included in:

```text
metadata/
├── mvtec.json
└── visa.json
```

Regenerating these two files is not required for reproducing the reported MVTec AD-to-VisA and VisA-to-MVTec AD experiments.

## Usage

Run metadata-generation scripts from the project root directory.

Most scripts define the dataset root path near the end of the corresponding Python file. Before running one of these scripts, replace the default dataset path with the location of the dataset in the local environment.

For example:

```bash
python generate_dataset_json/mvtec.py
```

Some scripts expose the dataset root through a command-line argument. Their available arguments can be inspected with:

```bash
python generate_dataset_json/oct17.py --help
python generate_dataset_json/brain_ad.py --help
python generate_dataset_json/kvasir.py --help
```

The generated JSON files contain relative image paths, relative mask paths, category names, defect types, and binary anomaly labels. The original dataset images are not included or redistributed.

## Br35H Preparation

`br35h_maskrcnn.py` additionally prepares the Br35H directory structure by processing source images and segmentation masks before generating metadata. The source, normal-image, and output paths defined in that script must be adjusted to the local environment before execution.

## Notes

Dataset archives may use different extracted directory names or layouts. Minor path adjustments may therefore be necessary before running the preprocessing scripts.

The generated metadata should be checked to ensure that all image and mask paths are relative to the corresponding dataset root.
