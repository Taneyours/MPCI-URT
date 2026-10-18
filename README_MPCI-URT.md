# MPCI-URT

This repository contains the code and a small demonstration dataset for the study:

**Penetrating the “black-box movement” in the urban rail transit network**

The original AFC datasets analyzed in this study cannot be publicly redistributed due to data-use and privacy restrictions. Therefore, we provide only an anonymized, feature-engineered demo dataset to illustrate the data format and computational workflow. The demo is intended for code execution and method demonstration, rather than reproduction of the full numerical results reported in the manuscript.

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

## Demo

Run the workflow from the repository root:

```bash
python scripts/build_trajectory_scores.py --config configs/demo.yaml
python scripts/train.py --config configs/demo.yaml
python scripts/evaluate.py --config configs/demo.yaml
```

The demo data are provided in `data/demo/`.
