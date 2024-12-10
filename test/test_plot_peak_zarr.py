#%%
import zarr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_chromosome_data(zarr_path: str, chromosome: str):
    """Load data for a specific chromosome from zarr store.
    
    Args:
        zarr_path: Path to zarr store
        chromosome: Chromosome name (e.g., 'chr1')
        
    Returns:
        Dictionary containing the data arrays
    """
    store = zarr.open(zarr_path, mode='r')
    return {
        'motifs': store['motifs'][chromosome][:],
        'hic': store['hic'][chromosome][:],
        'hic_oe': store['hic_oe'][chromosome][:],
        'peak_coords': store['peak_coords'][chromosome][:],
        'atac': store['atac'][chromosome][:]
    }

def plot_sample(data: dict, sample_idx: int, output_dir: Path):
    """Create visualization for a single sample.
    
    Args:
        data: Dictionary containing data arrays
        sample_idx: Index of the sample to plot
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots vertically arranged
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(6, 15), sharex=True, gridspec_kw={'height_ratios': [1, 1, 0.1, 0.1]})
    
    # Plot HiC O/E matrix
    sns.heatmap(data['hic_oe'][sample_idx], 
                cmap='coolwarm', 
                center=1,
                cbar=False,
                ax=ax1)
    ax1.set_title('HiC Observed/Expected Matrix')
    
    # Plot HiC matrix
    sns.heatmap(data['hic'][sample_idx], 
                cmap='YlOrRd', 
                cbar=False,
                ax=ax2)
    ax2.set_title('HiC Contact Matrix')
    
    # Plot ATAC signal
    atac_signal = data['atac'][sample_idx]
    ax3.plot(atac_signal, 'b-', linewidth=1)
    ax3.set_title('ATAC Signal')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Signal Strength')
    # plot ctcf signal
    motif_scores = data['motifs'][sample_idx]
    ax4.plot(motif_scores[:, 16], 'r-', linewidth=1)
    ax4.set_title('CTCF Motif Scores')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Score')
    
    # Add peak coordinates as text
    peak_coords = data['peak_coords'][sample_idx]
    plt.figtext(0.02, 0.02, 
                f'Peak coordinates: {peak_coords[0]}-{peak_coords[1]}',
                fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()
#%%
def main():
    """Main function to load and plot samples from zarr store."""
    # Configuration
    zarr_path = 'h1_esc_nucleotide_motif_adaptor_output_peak.zarr'
    output_dir = Path('peak_visualization')
    chromosome = 'chr1'  # Example chromosome
    n_samples = 5  # Number of samples to plot
    
    # Load data
    try:
        data = load_chromosome_data(zarr_path, chromosome)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Create visualizations
    print(f"Creating visualizations for {n_samples} samples from {chromosome}...")
    for i in range(min(n_samples, len(data['motifs']))):
        try:
            plot_sample(data, i, output_dir)
            print(f"Saved plot for sample {i}")
        except Exception as e:
            print(f"Error plotting sample {i}: {str(e)}")
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Number of samples: {len(data['motifs'])}")
    print(f"Motif vector size: {data['motifs'].shape[-1]}")
    print(f"HiC matrix size: {data['hic'].shape[1]}x{data['hic'].shape[2]}")
    print(f"ATAC signal length: {len(data['atac'][0])}")
#%%
main()
#%%
if __name__ == '__main__':
    main() 