# FAORU: Frequency-Adaptive Orthogonal Residual Updates

[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-orange)]()
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation of **"FAORU: Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks"** (Under review at TPAMI).

> **Note**: This paper is currently under peer review. The preprint will be made available upon acceptance.

## 🔥 Highlights

- **+2.78% Top-1** on ImageNet-1K with ViT-B (82.41% → 85.19%)
- **Frequency-Adaptive**: Per-frequency orthogonalization with learnable strengths λ_k
- **Plug-and-Play**: Drop-in replacement for standard residual connections
- **Theoretically Grounded**: Proven correlation attenuation & energy bounds
- **Universally Effective**: Works for CNNs (ResNet, ConvNeXt) and Transformers (ViT, Swin)

## 📰 News

- **[2024/11]** 📝 Paper under review at TPAMI
- **[2024/11]** 🚀 Code released for reproducibility

## 🎯 Key Results

| Model | Dataset | Baseline | FAORU-L | Gain | Config |
|-------|---------|----------|---------|------|--------|
| ViT-B | ImageNet-1K | 82.41% | **85.19%** | +2.78% | [config](configs/vit_b_imagenet.yaml) |
| ViT-B | CIFAR-100 | 81.28% | **84.83%** | +3.55% | [config](configs/vit_b_cifar100.yaml) |
| ResNet-50 | ImageNet-1K | 76.10% | **78.80%** | +2.70% | [config](configs/resnet50_imagenet.yaml) |
| ConvNeXt-B | ImageNet-1K | 83.80% | **86.50%** | +2.70% | [config](configs/convnext_b_imagenet.yaml) |

### Robustness Results
| Metric | Baseline | FAORU-L | Improvement |
|--------|----------|---------|-------------|
| ImageNet-C (mCE↓) | 79.8 | **74.2** | -5.6 |
| ImageNet-R (Acc↑) | 40.1% | **43.5%** | +3.4% |
| ImageNet-V2 (Acc↑) | 64.2% | **66.9%** | +2.7% |

## 🏗️ Architecture Overview

FAORU performs residual updates in the frequency domain with adaptive orthogonalization:

```
Input x → LayerNorm → f(x) → FFT → Per-Freq Orthogonalization → IFFT → Output
          ↓                                                                  ↑
          └──────────────────────── Identity Path ─────────────────────────┘
```

**Three-Stage Process:**
1. **Frequency Transform**: Apply FFT along feature dimension
2. **Adaptive Orthogonalization**: `F_⊥[k] = F[k] - λ_k · α_k · X[k]` for each frequency k
3. **Inverse Transform**: IFFT back to spatial domain

## 📦 Installation

### Requirements
```bash
Python >= 3.8
PyTorch >= 2.0.0
torchvision >= 0.15.0
timm >= 0.9.0
```

### Quick Install
```bash
# Clone the repository
git clone https://github.com/faoru-project/FAORU.git
cd FAORU

# Create conda environment
conda create -n faoru python=3.9
conda activate faoru

# Install dependencies
pip install -r requirements.txt

# Install FAORU
pip install -e .
```

## 🚀 Quick Start

### Using FAORU in Your Model

```python
import torch
from faoru import FAORUResidual

# Replace standard residual connection
class TransformerBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim)
        
        # Standard residual: x = x + attn(norm(x))
        # FAORU residual:
        self.residual = FAORUResidual(
            dim=dim,
            variant='learnable',  # 'piecewise', 'smooth', or 'learnable'
            epsilon=1e-6
        )
    
    def forward(self, x):
        # FAORU automatically handles: x + FAORU(attn(norm(x)))
        x = self.residual(x, self.attn(self.norm1(x)))
        return x
```

### Training from Scratch

```bash
# ImageNet-1K with ViT-B (8xH100)
torchrun --nproc_per_node=8 train.py \
    --config configs/vit_b_imagenet.yaml \
    --data-path /path/to/imagenet \
    --output outputs/vit_b_faoru

# CIFAR-100 with ViT-S (single GPU)
python train.py \
    --config configs/vit_s_cifar100.yaml \
    --data-path /path/to/cifar100 \
    --output outputs/vit_s_cifar100
```

### Evaluation with Pretrained Models

```bash
# Download pretrained model
wget https://github.com/faoru-project/FAORU/releases/download/v1.0/vit_b_faoru_imagenet.pth

# Evaluate
python validate.py \
    --model vit_base_patch16_224 \
    --checkpoint vit_b_faoru_imagenet.pth \
    --data-path /path/to/imagenet \
    --faoru-variant learnable
```

## 📊 Reproducing Paper Results

### Main Classification Results (Table 1)

```bash
# ImageNet-1K with ViT-B (5 seeds)
bash scripts/reproduce_main_results.sh

# This will run:
# - Standard Residual baseline
# - Spatial Orthogonal baseline  
# - FAORU-PC, FAORU-ST, FAORU-L
# with seeds {42, 1337, 2024, 3141, 9999}
```

### Ablation Studies (Table 4)

```bash
# Component ablation
bash scripts/ablation_components.sh

# Placement ablation
bash scripts/ablation_placement.sh

# Transform ablation (FFT vs DCT vs Hadamard)
bash scripts/ablation_transforms.sh
```

### Robustness Evaluation

```bash
# ImageNet-C
python eval_robustness.py \
    --benchmark imagenet-c \
    --checkpoint vit_b_faoru_imagenet.pth

# ImageNet-R
python eval_robustness.py \
    --benchmark imagenet-r \
    --checkpoint vit_b_faoru_imagenet.pth
```

## 📁 Project Structure

```
FAORU/
├── faoru/                      # Core FAORU implementation
│   ├── __init__.py
│   ├── residual.py            # FAORUResidual module
│   ├── variants.py            # λ_k parameterizations (PC/ST/L)
│   └── transforms.py          # FFT/DCT/Hadamard transforms
├── models/                     # Model architectures
│   ├── vit.py                 # Vision Transformer with FAORU
│   ├── resnet.py              # ResNet with FAORU
│   └── convnext.py            # ConvNeXt with FAORU
├── configs/                    # Training configurations
│   ├── vit_b_imagenet.yaml
│   ├── resnet50_imagenet.yaml
│   └── ...
├── scripts/                    # Reproduction scripts
│   ├── reproduce_main_results.sh
│   ├── ablation_components.sh
│   └── ...
├── tools/                      # Analysis tools
│   ├── visualize_lambda.py    # Visualize learned λ_k
│   ├── frequency_analysis.py  # Frequency spectrum analysis
│   └── export_onnx.py         # Export to ONNX
├── train.py                    # Training script
├── validate.py                 # Evaluation script
├── eval_robustness.py         # Robustness benchmarks
├── requirements.txt
└── setup.py
```

## 🔬 Analysis & Visualization

### Visualize Learned Frequency Strengths

```python
from tools.visualize_lambda import plot_lambda_curves

# Load checkpoint and visualize λ_k across layers
plot_lambda_curves(
    checkpoint='vit_b_faoru_imagenet.pth',
    layers=[2, 6, 10],  # Shallow, mid, deep
    output='lambda_curves.pdf'
)
```

### Frequency Spectrum Analysis

```python
from tools.frequency_analysis import analyze_frequency_distribution

# Analyze energy distribution across layers
analyze_frequency_distribution(
    model='vit_b_faoru',
    data_path='/path/to/imagenet/val',
    output='freq_analysis.pdf'
)
```

## 🎓 Citation

If you find FAORU useful in your research, please consider citing:

```bibtex
@article{faoru2024,
  title={FAORU: Frequency-Adaptive Orthogonal Residual Updates for Modern Vision Networks},
  author={Anonymous Authors},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  note={Under Review},
  year={2024}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [timm](https://github.com/huggingface/pytorch-image-models) for model implementations
- Inspired by orthogonal residual works: [OCNN](https://arxiv.org/abs/1911.12207), [Isometric Learning](https://arxiv.org/abs/2006.16992)
- Frequency analysis tools adapted from [FrequencyAnalysis](https://github.com/xxx/xxx)

## 📧 Contact

For questions and discussions, please:
- Open an issue on GitHub
- Email: [contact email to be added]

## 🔗 Related Projects

- [DeiT](https://github.com/facebookresearch/deit) - Vision Transformer training
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) - Modern CNN design
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch Image Models

---

**Note**: This is research code. We're actively working on optimizations and extensions. Contributions are welcome!
