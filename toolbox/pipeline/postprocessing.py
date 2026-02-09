from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["get_nms_map", "visualize_nms_process"]

def get_nms_map(best_pc: np.ndarray, best_theta_idx: np.ndarray, unique_thetas: np.ndarray) -> np.ndarray:
    """
    Thins the PC map by checking neighbors along the gradient direction.
    
    Args:
        best_pc: 2D array of max Phase Congruency values
        best_theta_idx: 2D array of indices pointing to the winning orientation
        unique_thetas: 1D array of angles used in the filter bank
    """
    H, W = best_pc.shape
    nms_pc = np.zeros_like(best_pc)
    
    # Pre-calculate direction offsets to avoid repeated sin/cos calls in the loop
    dr_lookup = np.round(np.sin(unique_thetas)).astype(int)
    dc_lookup = np.round(np.cos(unique_thetas)).astype(int)
    
    # Iterate through the image (avoiding borders)
    for r in range(1, H-1):
        for c in range(1, W-1):
            t_idx = best_theta_idx[r, c]
            dr = dr_lookup[t_idx]
            dc = dc_lookup[t_idx]
            
            val = best_pc[r, c]
            
            # Non-Maximum Suppression:
            # Check if the current pixel is a local maximum along the gradient direction
            if val >= best_pc[r + dr, c + dc] and val >= best_pc[r - dr, c - dc]:
                nms_pc[r, c] = val
                
    return nms_pc

def visualize_nms_process(best_pc: np.ndarray, nms_pc: np.ndarray, threshold: float = 0.05, figsize: tuple = (16, 8)):
    """
    Compares the raw PC map with the NMS-thinned map.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)
    
    # Left: Raw Phase Congruency
    im0 = axes[0].imshow(best_pc, cmap='magma')
    axes[0].set_title("Raw Phase Congruency (Thick)")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Right: NMS Result
    # Using the threshold to highlight the skeleton
    im1 = axes[1].imshow(nms_pc > threshold, cmap='gray_r') 
    axes[1].set_title(f"NMS Map (Skeleton, T > {threshold})")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()