## Bandwidth Selection for KDE-based Calibration Error Estimation

A PyTorch library implementing **automatic bandwidth selection methods** for Kernel Density Estimation (KDE) in model calibration assessment.

### Overview

Bandwidth selection is critical for KDE-based calibration error estimation. This repository provides:

- **MLE-LOO**: Maximum Likelihood Estimation with Leave-One-Out 
- **Risk-LOO**: Risk-based bandwidth selection that minimizes squared error between KDE estimates and empirical risk

Both methods automatically select optimal bandwidth(s) from a candidate grid, with support for per-class bandwidth in class-wise calibration mode.

#### Key Features
- Per-class bandwidth for class-wise calibration
- Support for multiple calibration modes: `binary`, `canonical`, `classwise`
- Support for L2 and KL divergence metrics

### Installation

```bash
pip install torch numpy matplotlib
```

### Quick Start

#### Bandwidth Selection

```python
import torch
from kde.bandwidth import select_bandwidth

# f: predicted probabilities (N, K)
# y: ground truth labels (N,)
f = torch.softmax(logits, dim=1)
y = labels

# Select bandwidth using Risk-LOO method
bw = select_bandwidth(f, y, mode="classwise", ce_type="l2", method="risk-loo")
# Returns per-class bandwidths for classwise mode

# Select bandwidth using MLE-LOO method
bw = select_bandwidth(f, y, mode="classwise", ce_type="l2", method="MLE-loo")
```

#### Compute Calibration Error with Selected Bandwidth

```python
from kde.kde_estimators import get_ece_kde

# Use the selected bandwidth
ce = get_ece_kde(f, y, bw, mode="classwise", ce_type="l2")
```

#### Bandwidth Selection Methods

| Method | Description |
|--------|-------------|
| `MLE-loo` | Maximizes Leave-One-Out log-likelihood of the KDE |
| `risk-loo` | Minimizes squared error between KDE risk estimate and empirical risk |

#### Calibration Modes

| Mode | Description |
|------|-------------|
| `binary` | Binary calibration (top-class confidence) |
| `canonical` | Canonical calibration (full probability vector) |
| `classwise` | Class-wise calibration (per-class bandwidths) |

#### Running Experiments

```bash
python demo.py --mode classwise --ce_type l2 --num_runs 3
```

Results are saved to `figs/bandwidth/`:
- Bandwidth convergence plots across sample sizes
- Calibration curves and reliability diagrams
