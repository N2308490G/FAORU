# FAORU: Frequency-Adaptive Orthogonal Residual Updates

[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official repository for **"FAORU: Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks"** (Under review at TPAMI).

---

## 📢 Important Notice: Code Release Policy

**The complete implementation will be released after the paper is officially published.**

This repository currently maintains the project structure and documentation. Full source code, pre-trained models, and reproduction scripts will be made available immediately upon publication in TPAMI.

---

## Why Code Is Not Yet Available

In academic research, code release is subject to institutional policies and research integrity considerations. Our university's research office has established clear guidelines:

### 1. **Academic Integrity and Research Protection**

Before formal publication, research has not undergone complete peer review. Premature code release could lead to:
- Unauthorized use, modification, or misinterpretation
- Compromise of research originality and credibility
- Risks to ongoing peer review integrity

### 2. **Intellectual Property and Priority Rights**

This work involves novel algorithms (frequency-adaptive orthogonalization with learnable strengths). Early release could:
- Impact priority recognition in the field
- Lead to intellectual property disputes
- Affect proper academic attribution

### 3. **Review Process Requirements**

Peer review may require experimental modifications. Pre-publication release could:
- Create version discrepancies
- Affect research reproducibility
- Violate journal pre-publication policies

### 4. **Institutional Regulations**

University policies govern research code management, especially for funded projects, requiring compliance with institutional standards.

---

## What Will Be Released

Upon official publication, we will immediately release:

### ✅ Complete Implementation
- Core FAORU modules (`residual.py`, `variants.py`, `transforms.py`)
- Model integrations (ViT, ResNet, ConvNeXt, Swin)
- Training and evaluation scripts
- Analysis and visualization tools

### ✅ Pre-trained Models
- Checkpoint files for all reported results
- ImageNet-1K trained weights
- Robustness-enhanced models

### ✅ Comprehensive Documentation
- Installation instructions
- Usage tutorials and examples
- API documentation
- Experiment reproduction guides

### ✅ Reproducibility Materials
- Complete training configurations
- Hyperparameter settings
- Benchmark evaluation protocols
- Dataset preparation scripts

---

## Key Results (From Paper)

### ImageNet-1K Classification

| Model | Baseline | FAORU-L | Gain |
|-------|----------|---------|------|
| ViT-B | 82.41% | **85.19%** | +2.78% |
| ViT-S | 80.23% | **82.87%** | +2.64% |
| ResNet-50 | 76.10% | **78.80%** | +2.70% |

### Robustness Improvements

| Benchmark | Baseline | FAORU-L | Gain |
|-----------|----------|---------|------|
| ImageNet-C (mCE↓) | 79.8 | **74.2** | -5.6 |
| ImageNet-R (Acc↑) | 40.1% | **43.5%** | +3.4% |
| ImageNet-V2 (Acc↑) | 64.2% | **66.9%** | +2.7% |

---

## Method Overview

FAORU performs frequency-adaptive orthogonal residual updates:

```
x → f(x) → FFT → Per-Frequency Orthogonalization → IFFT → Output
↓                                                           ↑
└─────────────────── Identity Path ───────────────────────┘
```

**Three-Stage Process:**
1. **Frequency Transform**: FFT along feature dimension
2. **Adaptive Orthogonalization**: `F_⊥[k] = F[k] - λ_k · α_k · X[k]`
3. **Inverse Transform**: IFFT back to spatial domain

**Key Innovation**: Per-frequency learnable strengths λ_k adapt orthogonalization across the frequency spectrum.

---

## Repository Structure

Current structure (code to be released):

```
FAORU/
├── faoru/              # Core implementation
│   ├── residual.py     # FAORU residual module
│   ├── variants.py     # Lambda parameterizations
│   └── transforms.py   # Frequency transforms
├── models/             # Model integrations
│   ├── vit.py         # Vision Transformers
│   ├── resnet.py      # ResNets
│   └── convnext.py    # ConvNeXt
├── tools/              # Analysis & visualization
├── configs/            # Training configurations
├── scripts/            # Reproduction scripts
├── tests/              # Unit tests
├── README.md           # This file
├── CONTRIBUTING.md     # Guidelines
├── LICENSE             # MIT License
└── requirements.txt    # Dependencies
```

---

## Installation (After Release)

Once code is released:

```bash
# Clone repository
git clone https://github.com/N2308490G/FAORU.git
cd FAORU

# Install dependencies
pip install -r requirements.txt

# Install FAORU
pip install -e .
```

**Requirements**: Python ≥ 3.8, PyTorch ≥ 2.0.0, timm ≥ 0.9.0

---

## Usage Example (After Release)

```python
from faoru import FAORUResidual
from models import vit_base_faoru

# Create FAORU-enhanced ViT
model = vit_base_faoru(
    pretrained=True,
    faoru_variant='learnable',
    faoru_transform='fft'
)

# Or integrate FAORU into custom models
faoru_residual = FAORUResidual(
    dim=768,
    variant='learnable',
    transform_type='fft'
)
```

---

## Citation

If you find this work useful, please cite (once published):

```bibtex
@article{faoru2024,
  title={FAORU: Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks},
  author={Anonymous Authors},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note={Under Review},
  year={2024}
}
```

---

## Timeline

- **[2024/11]** Paper submitted to TPAMI
- **[2025/11]** Repository structure created
- **[TBD]** Paper acceptance notification
- **[TBD]** **Complete code release immediately upon publication**

---

## Contact

For inquiries about the paper or code release:
- Watch this repository for updates
- Open an issue for questions

We appreciate your understanding and support in maintaining research integrity. We are committed to open science and will release all materials promptly upon publication.

---

**Expected Code Release**: Immediately upon paper acceptance and publication

**License**: MIT License (permissive open-source)

**Last Updated**: November 2025
