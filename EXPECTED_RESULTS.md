# Expected Results

The following values are representative results obtained with the
PE-Core-L14-336 backbone at an input resolution of 518 × 518.

## MVTec AD to VisA

| Dataset | I-AUC | P-AUC |  PRO |
| ------- | ----: | ----: | ---: |
| VisA    |  90.4 |  95.7 | 93.0 |

## VisA to MVTec AD

| Dataset  | I-AUC | P-AUC |  PRO |
| -------- | ----: | ----: | ---: |
| MVTec AD |  93.7 |  90.9 | 88.1 |

The reported values are obtained from a single run with the fixed random
seed specified in the training and evaluation code.

Minor numerical differences may arise from the CUDA version, GPU
architecture, and nondeterministic CUDA operators.
