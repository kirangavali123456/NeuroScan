import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_anomaly_heatmap(original, reconstructed):
    """
    Calculates the absolute difference between input and output.
    Returns the difference map (heatmap).
    """
    # Move to CPU and numpy
    orig_np = original.squeeze().cpu().detach().numpy()
    recon_np = reconstructed.squeeze().cpu().detach().numpy()
    
    # Calculate absolute difference (The "Anomaly Signal")
    diff_map = np.abs(orig_np - recon_np)
    return orig_np, recon_np, diff_map

def plot_anomaly_detection(orig, recon, diff, threshold=0.1):
    """
    Visualizes the Medical Anomaly Detection process.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original Image
    axes[0].imshow(orig, cmap='gray')
    axes[0].set_title("Input (X-Ray/Scan)")
    axes[0].axis('off')
    
    # 2. Reconstructed Image (What the model 'thinks' it should be)
    axes[1].imshow(recon, cmap='gray')
    axes[1].set_title("VAE Reconstruction (Healthy Model)")
    axes[1].axis('off')
    
    # 3. Anomaly Map (Difference)
    # We use a 'hot' colormap to highlight errors (tumors/anomalies)
    im = axes[2].imshow(diff, cmap='inferno', vmin=0, vmax=1)
    axes[2].set_title("Anomaly Detection Map")
    axes[2].axis('off')
    
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    return fig
