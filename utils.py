import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score

def get_anomaly_heatmap(original, reconstructed):
    orig_np = original.squeeze().cpu().detach().numpy()
    recon_np = reconstructed.squeeze().cpu().detach().numpy()
    diff_map = np.abs(orig_np - recon_np)
    return orig_np, recon_np, diff_map

def calculate_metrics(model, test_loader, threshold_percentile=95):
    model.eval()
    errors = []
    labels = []
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for data, label in test_loader:
            data = data.to(next(model.parameters()).device)
            if hasattr(model, 'reparameterize'):
                recon, _, _ = model(data)
            else:
                recon = model(data)
            loss = criterion(recon, data).mean(dim=[1, 2, 3])
            errors.extend(loss.cpu().numpy())
            labels.extend(label.numpy())
            
    threshold = np.percentile(errors, threshold_percentile)
    preds = [1 if e > threshold else 0 for e in errors]
    
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='binary')
    return acc, f1

# --- UPDATED PLOTTING FUNCTION ---
def plot_comparison(original, recons, titles):
    """
    Plots a 2-row grid:
    Row 1: Original + Reconstructions
    Row 2: Difference Maps (Heatmaps)
    """
    num_models = len(recons)
    cols = num_models + 1
    rows = 2
    
    # Taller figure to accommodate 2 rows
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 8))
    
    # Helper to handle 1D axes array if only 1 model is passed
    if num_models == 0: return fig
    if len(axes.shape) == 1: axes = axes.reshape(rows, cols)

    # Prepare Original
    orig_np = original.squeeze().cpu().detach().numpy()
    
    # --- ROW 1: RECONSTRUCTIONS ---
    # Col 0: Original
    axes[0, 0].imshow(orig_np, cmap='gray')
    axes[0, 0].set_title("Original Input", fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Models
    for i, (recon, title) in enumerate(zip(recons, titles)):
        recon_np = recon.squeeze().cpu().detach().numpy()
        axes[0, i+1].imshow(recon_np, cmap='gray')
        axes[0, i+1].set_title(f"{title}\n(Reconstruction)", fontsize=12)
        axes[0, i+1].axis('off')

    # --- ROW 2: DIFFERENCE MAPS ---
    # Col 0: Blank (or Legend)
    axes[1, 0].text(0.5, 0.5, "Difference Maps\n(Anomaly Detection)", 
                    ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')

    # Models
    for i, (recon, title) in enumerate(zip(recons, titles)):
        recon_np = recon.squeeze().cpu().detach().numpy()
        # Calculate Difference
        diff_np = np.abs(orig_np - recon_np)
        
        # Plot Heatmap (Inferno makes anomalies glow bright orange/yellow)
        im = axes[1, i+1].imshow(diff_np, cmap='inferno', vmin=0, vmax=1)
        axes[1, i+1].set_title(f"{title}\n(Anomaly Map)", fontsize=12, color='red')
        axes[1, i+1].axis('off')

    plt.tight_layout()
    return fig
