"""
Frequency-Domain Energy Analysis

Analyzes energy distribution across frequency bands before/after FAORU,
reproducing frequency statistics from the paper.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import argparse
import pandas as pd

sns.set_style("whitegrid")


def analyze_frequency_energy(
    features: torch.Tensor,
    num_bands: int = 4
) -> Dict[str, float]:
    """
    Compute energy distribution across frequency bands.
    
    Args:
        features: [B, S, D] input features
        num_bands: Number of frequency bands to split into
    
    Returns:
        dict: {
            'low': energy in low frequencies,
            'mid': energy in mid frequencies,
            'high': energy in high frequencies,
            'total': total energy
        }
    """
    B, S, D = features.shape
    
    # Apply FFT
    F = torch.fft.rfft(features, dim=-1)  # [B, S, D//2+1]
    
    # Energy: |F[k]|²
    energy = (F.real**2 + F.imag**2).mean(dim=(0, 1))  # [D//2+1]
    
    total_energy = energy.sum().item()
    freq_dim = energy.shape[0]
    
    # Split into bands
    band_size = freq_dim // num_bands
    band_energies = {}
    
    if num_bands == 4:
        # Low: [0, 0.25], Mid-Low: [0.25, 0.5], Mid-High: [0.5, 0.75], High: [0.75, 1.0]
        bands = {
            'low': (0, band_size),
            'mid_low': (band_size, 2*band_size),
            'mid_high': (2*band_size, 3*band_size),
            'high': (3*band_size, freq_dim)
        }
    else:
        # Simple low/mid/high split
        bands = {
            'low': (0, freq_dim // 3),
            'mid': (freq_dim // 3, 2 * freq_dim // 3),
            'high': (2 * freq_dim // 3, freq_dim)
        }
    
    for band_name, (start, end) in bands.items():
        band_energy = energy[start:end].sum().item()
        band_energies[band_name] = band_energy
        band_energies[f'{band_name}_ratio'] = band_energy / total_energy * 100
    
    band_energies['total'] = total_energy
    
    return band_energies


def compare_before_after_faoru(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 1000,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Compare frequency energy distribution before/after FAORU.
    
    Returns:
        DataFrame with columns: layer, band, before, after, change
    """
    model.eval()
    model.to(device)
    
    results = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(dataloader):
            if batch_idx * images.size(0) >= num_samples:
                break
            
            images = images.to(device)
            
            # Hook to capture features
            features_dict = {}
            
            def hook_fn(name):
                def fn(module, input, output):
                    # Capture before FAORU (input to residual)
                    if hasattr(output, 'freq_analysis'):
                        # FAORU module with analysis
                        features_dict[f'{name}_before'] = input[0].detach().cpu()
                        features_dict[f'{name}_after'] = output.detach().cpu()
                return fn
            
            # Register hooks on FAORU residual modules
            hooks = []
            for name, module in model.named_modules():
                if 'residual' in name and hasattr(module, 'lambda_module'):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
            
            # Forward pass
            _ = model(images)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Analyze captured features
            for key in features_dict:
                if '_before' in key:
                    layer_name = key.replace('_before', '')
                    before = features_dict[key]
                    after = features_dict.get(f'{layer_name}_after', None)
                    
                    if after is not None:
                        stats_before = analyze_frequency_energy(before)
                        stats_after = analyze_frequency_energy(after)
                        
                        for band in ['low', 'mid_low', 'mid_high', 'high']:
                            if f'{band}_ratio' in stats_before:
                                results.append({
                                    'layer': layer_name,
                                    'band': band,
                                    'before': stats_before[f'{band}_ratio'],
                                    'after': stats_after[f'{band}_ratio'],
                                    'change': stats_after[f'{band}_ratio'] - stats_before[f'{band}_ratio']
                                })
    
    return pd.DataFrame(results)


def plot_frequency_energy_bars(
    df: pd.DataFrame,
    layers: list,
    output_path: str = 'frequency_energy.pdf'
):
    """
    Bar plot comparing before/after frequency energy.
    """
    fig, axes = plt.subplots(1, len(layers), figsize=(5*len(layers), 5))
    
    if len(layers) == 1:
        axes = [axes]
    
    for ax, layer in zip(axes, layers):
        layer_data = df[df['layer'].str.contains(str(layer))]
        
        if layer_data.empty:
            continue
        
        # Average over different residuals in the layer
        avg_data = layer_data.groupby('band').mean().reset_index()
        
        x = np.arange(len(avg_data))
        width = 0.35
        
        ax.bar(x - width/2, avg_data['before'], width, 
               label='Before FAORU', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, avg_data['after'], width,
               label='After FAORU', alpha=0.8, color='coral')
        
        ax.set_xlabel('Frequency Band', fontsize=12)
        ax.set_ylabel('Energy Ratio (%)', fontsize=12)
        ax.set_title(f'Layer {layer}', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(avg_data['band'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def plot_frequency_spectrum(
    features_before: torch.Tensor,
    features_after: torch.Tensor,
    layer_name: str = 'Layer 6',
    output_path: str = 'frequency_spectrum.pdf'
):
    """
    Plot full frequency spectrum before/after FAORU.
    """
    # Average over batch and sequence
    F_before = torch.fft.rfft(features_before, dim=-1)  # [B, S, D//2+1]
    F_after = torch.fft.rfft(features_after, dim=-1)
    
    energy_before = (F_before.real**2 + F_before.imag**2).mean(dim=(0, 1)).numpy()
    energy_after = (F_after.real**2 + F_after.imag**2).mean(dim=(0, 1)).numpy()
    
    freq_dim = energy_before.shape[0]
    normalized_freq = np.arange(freq_dim) / (freq_dim * 2)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear scale
    ax1.plot(normalized_freq, energy_before, label='Before FAORU', linewidth=2, alpha=0.8)
    ax1.plot(normalized_freq, energy_after, label='After FAORU', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Normalized Frequency', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1.set_title(f'{layer_name} - Frequency Spectrum (Linear)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(normalized_freq, energy_before, label='Before FAORU', linewidth=2, alpha=0.8)
    ax2.semilogy(normalized_freq, energy_after, label='After FAORU', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Normalized Frequency', fontsize=12)
    ax2.set_ylabel('Energy (log scale)', fontsize=12)
    ax2.set_title(f'{layer_name} - Frequency Spectrum (Log)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to: {output_path}")
    plt.close()


def compute_orthogonality_metric(
    F: torch.Tensor,
    X: torch.Tensor
) -> float:
    """
    Compute orthogonality metric: cos(θ) = <F, X> / (||F|| ||X||)
    
    Args:
        F: [B, S, D//2+1] frequency-domain features (complex)
        X: [B, S, D//2+1] frequency-domain skip connection (complex)
    
    Returns:
        Average absolute cosine similarity (0 = orthogonal, 1 = parallel)
    """
    # Inner product in frequency domain
    inner_prod = (F.real * X.real + F.imag * X.imag).sum(dim=-1)  # [B, S]
    
    # Norms
    norm_F = torch.sqrt((F.real**2 + F.imag**2).sum(dim=-1))  # [B, S]
    norm_X = torch.sqrt((X.real**2 + X.imag**2).sum(dim=-1))
    
    # Cosine similarity
    cos_sim = inner_prod / (norm_F * norm_X + 1e-8)
    
    return cos_sim.abs().mean().item()


def main():
    parser = argparse.ArgumentParser(description='Frequency domain analysis')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to ImageNet validation data')
    parser.add_argument('--layers', type=int, nargs='+', default=[2, 6, 10],
                       help='Layers to analyze')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to analyze')
    parser.add_argument('--output-dir', type=str, default='./visualizations',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading model...")
    # TODO: Load actual model from checkpoint
    # model = load_model(args.checkpoint)
    
    print("Loading data...")
    # TODO: Load validation dataloader
    # dataloader = create_dataloader(args.data_dir, batch_size=64)
    
    print("\nAnalyzing frequency energy distribution...")
    # df = compare_before_after_faoru(model, dataloader, args.num_samples)
    # df.to_csv(output_dir / 'frequency_stats.csv', index=False)
    # print(f"Saved stats to: {output_dir / 'frequency_stats.csv'}")
    
    print("\nGenerating visualizations...")
    # plot_frequency_energy_bars(df, args.layers, output_dir / 'frequency_energy.pdf')
    
    print("\nPlaceholder: Implement model loading and dataloader creation")


if __name__ == '__main__':
    main()
