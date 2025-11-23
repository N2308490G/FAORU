"""
Visualize Learned Frequency Strengths (λ_k)

Creates publication-quality plots of learned λ_k curves across layers,
reproducing Figure 3 from the paper.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import argparse

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def load_lambda_from_checkpoint(checkpoint_path: str) -> Dict[str, np.ndarray]:
    """
    Extract λ_k values from all FAORU layers in checkpoint.
    
    Returns:
        dict: {layer_name: lambda_k_array}
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    lambda_dict = {}
    
    for key, value in state_dict.items():
        if 'residual' in key and 'lambda_module.weights' in key:
            # Extract layer index from key
            # e.g., "blocks.2.residual1.lambda_module.weights"
            parts = key.split('.')
            layer_idx = int(parts[1])
            residual_idx = parts[2]  # 'residual1' or 'residual2'
            
            # Apply sigmoid to get actual λ_k values
            lambda_k = torch.sigmoid(value).numpy()
            
            layer_name = f"Layer {layer_idx} ({residual_idx})"
            lambda_dict[layer_name] = lambda_k
    
    return lambda_dict


def plot_lambda_curves(
    checkpoint_path: str,
    layers: List[int],
    output_path: str = 'lambda_curves.pdf',
    title: str = 'Learned Frequency-Adaptive Strengths'
):
    """
    Plot λ_k curves for selected layers.
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        layers: List of layer indices to plot (e.g., [2, 6, 10])
        output_path: Where to save the figure
        title: Plot title
    """
    lambda_dict = load_lambda_from_checkpoint(checkpoint_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, layer_idx in enumerate(layers):
        # Find lambda for this layer (use residual1 for attention)
        key = f"Layer {layer_idx} (residual1)"
        
        if key not in lambda_dict:
            print(f"Warning: {key} not found in checkpoint")
            continue
        
        lambda_k = lambda_dict[key]
        D = len(lambda_k)
        
        # Normalized frequency: k/D ∈ [0, 0.5] for rFFT
        normalized_freq = np.arange(D) / (D * 2)
        
        # Plot with error band (simulated from single checkpoint)
        ax.plot(
            normalized_freq, 
            lambda_k, 
            label=f'Layer {layer_idx}',
            color=colors[i % len(colors)],
            linewidth=2.5,
            alpha=0.9
        )
    
    ax.axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.3, color='gray', linestyle=':', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Normalized Frequency (k/D)', fontsize=14)
    ax.set_ylabel('λ_k (Orthogonalization Strength)', fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 0.5])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_lambda_heatmap(
    checkpoint_path: str,
    output_path: str = 'lambda_heatmap.pdf',
    title: str = 'Layer-wise λ_k Distribution'
):
    """
    Create heatmap of λ_k across all layers and frequencies.
    """
    lambda_dict = load_lambda_from_checkpoint(checkpoint_path)
    
    # Organize into matrix: [num_layers, freq_dim]
    layer_names = sorted([k for k in lambda_dict.keys() if 'residual1' in k],
                        key=lambda x: int(x.split()[1]))
    
    if not layer_names:
        print("No layers found!")
        return
    
    lambda_matrix = np.array([lambda_dict[name] for name in layer_names])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    im = ax.imshow(
        lambda_matrix,
        aspect='auto',
        cmap='viridis',
        vmin=0,
        vmax=1,
        interpolation='bilinear'
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('λ_k', rotation=270, labelpad=20, fontsize=14)
    
    # Labels
    ax.set_xlabel('Frequency Index (k)', fontsize=14)
    ax.set_ylabel('Layer', fontsize=14)
    ax.set_title(title, fontsize=16, pad=15)
    
    # Y-axis: layer indices
    layer_indices = [int(name.split()[1]) for name in layer_names]
    ax.set_yticks(range(len(layer_indices)))
    ax.set_yticklabels(layer_indices)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_lambda_statistics(
    checkpoint_path: str,
    output_path: str = 'lambda_stats.pdf'
):
    """
    Plot statistics of λ_k: mean, std, min, max per layer.
    """
    lambda_dict = load_lambda_from_checkpoint(checkpoint_path)
    
    layer_names = sorted([k for k in lambda_dict.keys() if 'residual1' in k],
                        key=lambda x: int(x.split()[1]))
    
    layer_indices = [int(name.split()[1]) for name in layer_names]
    means = [lambda_dict[name].mean() for name in layer_names]
    stds = [lambda_dict[name].std() for name in layer_names]
    mins = [lambda_dict[name].min() for name in layer_names]
    maxs = [lambda_dict[name].max() for name in layer_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean ± Std
    ax1.plot(layer_indices, means, 'o-', label='Mean', linewidth=2, markersize=8)
    ax1.fill_between(
        layer_indices,
        np.array(means) - np.array(stds),
        np.array(means) + np.array(stds),
        alpha=0.3,
        label='± Std'
    )
    ax1.set_xlabel('Layer', fontsize=14)
    ax1.set_ylabel('λ_k', fontsize=14)
    ax1.set_title('Mean λ_k Across Layers', fontsize=16)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Min/Max Range
    ax2.fill_between(layer_indices, mins, maxs, alpha=0.4, label='Range')
    ax2.plot(layer_indices, means, 'r-', label='Mean', linewidth=2)
    ax2.set_xlabel('Layer', fontsize=14)
    ax2.set_ylabel('λ_k', fontsize=14)
    ax2.set_title('λ_k Range Across Layers', fontsize=16)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize learned λ_k')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 6, 10],
                       help='Layers to plot (default: 2 6 10)')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    
    # 1. Lambda curves
    print("\n[1/3] Plotting λ_k curves...")
    plot_lambda_curves(
        args.checkpoint,
        args.layers,
        output_dir / 'lambda_curves.pdf'
    )
    
    # 2. Lambda heatmap
    print("[2/3] Creating λ_k heatmap...")
    plot_lambda_heatmap(
        args.checkpoint,
        output_dir / 'lambda_heatmap.pdf'
    )
    
    # 3. Lambda statistics
    print("[3/3] Plotting λ_k statistics...")
    plot_lambda_statistics(
        args.checkpoint,
        output_dir / 'lambda_stats.pdf'
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
