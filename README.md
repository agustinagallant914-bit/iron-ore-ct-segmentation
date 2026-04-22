# iron-ore-ct-segmentation

## Overview

This repository contains code for preprocessing, training, and evaluation of a dual-stream model for iron ore pellet CT image segmentation.

## Dataset

The dataset is available at Zenodo:
https://doi.org/10.5281/zenodo.19688595

## Files

* `Preprocessing.py` – data preprocessing
* `Dual.py` – model training and evaluation


## Usage

### 1. Preprocess data

```
python Preprocessing.py
```

### 2. Train and test

```
python Dual.py
```

## Output

* Trained model (`.pth`)
* Evaluation results (IoU, Dice and others )

## Code availability

All code used in this study is provided in this repository.

