# Dataset Metadata

For reproducibility, this directory provides the metadata JSON files used in our cross-dataset experiments:

```text
metadata/
├── mvtec.json
└── visa.json
```

The metadata files contain the training and test splits for MVTec AD and VisA. Each sample is represented by the following fields:

* `img_path`: image path relative to the corresponding dataset root;
* `mask_path`: anomaly mask path relative to the dataset root, or an empty string for normal samples;
* `cls_name`: object or texture category;
* `specie_name`: defect type when available;
* `anomaly`: binary anomaly label, where `0` denotes normal and `1` denotes anomalous.

These files contain only relative file paths, category information, anomaly labels, and mask paths. They do not contain or redistribute any original dataset images.

The provided metadata files correspond to the dataset organization used in the reported experiments. Users should set the dataset root paths to the local locations of MVTec AD and VisA when running the training or evaluation scripts.

The scripts under `generate_dataset_json/` are provided as preprocessing references for regenerating metadata. Differences in dataset releases, extracted directory names, or local organization may result in metadata files with different paths or sample ordering.
