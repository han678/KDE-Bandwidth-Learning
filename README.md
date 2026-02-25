# Calibration Error Estimation with KDE

A PyTorch library for estimating calibration error in machine learning models using Kernel Density Estimation (KDE) and binning methods.

## Overview

This repository provides implementations of:
- **KDE-based calibration error estimators** with automatic bandwidth selection (MLE, risk-aware)
- **Binning-based calibration error estimators** (equal-width, adaptive)
- Support for **multiple calibration modes**: binary, canonical, and class-wise
- Support for **multiple error metrics**: L2 and KL divergence

## Installation

```bash
pip install torch numpy matplotlib
```

```

### Running the Demo

```bash
python demo.py --mode classwise --ce_type l2 --num_runs 3
```
Results and plots are saved to the `figs/` directory:
- `figs/bandwidth/` - Bandwidth convergence plots
- Calibration curves and reliability diagrams

### Using KDE Estimators

```python
import torch
from kde.kde_estimators import get_ece_kde

# f: predicted probabilities (N, K)
# y: ground truth labels (N,)
f = torch.softmax(logits, dim=1)
y = labels

# Compute calibration error with KDE
bw = 0.1  # bandwidth
ce = get_ece_kde(f, y, bw, mode="classwise", ce_type="l2")
```

### Using Binning Estimators

```python
from bin_method import get_ece_bin

# Compute calibration error with equal-width binning
ce = get_ece_bin(f, y, n_bins=20, mode="classwise", ce_type="l2")
```

## Calibration Modes

| Mode | Description |
|------|-------------|
| `binary` | Binary calibration (top-class confidence) |
| `canonical` | Canonical calibration (full probability vector) |
| `classwise` | Class-wise calibration (per-class probabilities) |

## Error Metrics

| Metric | Description |
|--------|-------------|
| `l2` | Squared L2 distance |
| `kl` | KL divergence |