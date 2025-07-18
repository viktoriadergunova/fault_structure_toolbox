"""
This module has three functions:

* `create_gabor_kernels()` – generate kernel bank (psi = π → dark‑on‑bright).
* `run()`                   – return `(mask, orientation, magnitude)` arrays.
* `debug_tiles()`           – **optional** per‑tile visual debug inside
                              notebooks; draws gray tile / mask / overlay.

all three can be imported via:
```python
from toolbox.pipeline.gabor import run, debug_tiles
```
"""


from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt  # only used by debug_tiles

__all__ = ["create_gabor_kernels", "run", "debug_tiles"]

# ----------------------------------------------------------------------------
# Gabor kernel setup 
# ----------------------------------------------------------------------------

def create_gabor_kernels() -> tuple[list[np.ndarray], np.ndarray]:
    """Return a list of Gabor kernels (psi = π) plus their orientations (deg)."""

    kernels: list[np.ndarray] = []
    orientations = np.arange(0, 180, 16)
    for theta_deg in orientations:
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel(
            (101, 101),
            sigma=11,
            theta=theta,
            lambd=25,
            gamma=0.2,
            psi=np.pi,  # psi = π → polarity: positive resp = dark‑on‑bright
        )
        kernels.append(kernel)
    return kernels, orientations

# ----------------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------------

def run(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,  #only if qgis boundary artifacts
    percentile: int = 90,
):
    """Process an RGB image and return mask, orientation map, and magnitude."""

    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    gabor_mask = np.zeros((height, width), dtype=np.uint8)
    gabor_orientation = np.zeros((height, width), dtype=np.uint8)
    gabor_magnitude = np.zeros((height, width), dtype=np.float32)

    kernels, _ = create_gabor_kernels()
    stride = tile_size  # 0 % overlap

    # -----------------------------------------------------------------
    # Tile loop
    # -----------------------------------------------------------------
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            # Ignore tiny edge tiles (<50 % of tile_size)
            if (y_end - y) < tile_size // 2 or (x_end - x) < tile_size // 2:
                continue

            rgb_tile = rgb[y:y_end, x:x_end]

            # Local contrast normalisation
            rgb_tile = cv2.normalize(rgb_tile, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
            gray_norm = gray_tile.astype(np.float32) / 255.0

            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray_enh = clahe.apply((gray_norm * 255).astype(np.uint8))
            gray_enh = gray_enh.astype(np.float32) / 255.0

            # Gabor filtering (signed responses retained)
            responses = [cv2.filter2D(gray_norm, cv2.CV_32F, k) for k in kernels]
            stack = np.stack(responses, axis=-1)

            # keep the *maximum positive* response; negatives are supressed –
            # this enforces the polarity (dark‑on‑bright).
            #  use np.abs(stack) -> if both polarities are needed
        
            max_response = np.max(stack, axis=-1)

            # Normalise within tile
            if max_response.max() > max_response.min():
                max_response = (max_response - max_response.min()) / (
                    max_response.max() - max_response.min()
                )

            thresh = np.percentile(max_response, percentile)
            tile_mask = (max_response > thresh).astype(np.uint8)
            tile_orient = np.argmax(stack, axis=-1)

            # Write back to global arrays
            gabor_mask[y:y_end, x:x_end] = np.maximum(
                gabor_mask[y:y_end, x:x_end], tile_mask
            )
            gabor_orientation[y:y_end, x:x_end] = np.where(
                tile_mask > 0, tile_orient, gabor_orientation[y:y_end, x:x_end]
            )
            gabor_magnitude[y:y_end, x:x_end] = np.maximum(
                gabor_magnitude[y:y_end, x:x_end], max_response
            )

    return gabor_mask, gabor_orientation, gabor_magnitude

# ----------------------------------------------------------------------------
#Tile Visualizer
# ----------------------------------------------------------------------------

def debug_tiles(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,
    percentile: int = 90,
):
    """Iterate through tiles and display (gray, mask, overlay).

    prints each tile for for visual inspection.
    
    """

    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    h, w, _ = rgb.shape
    kernels, _ = create_gabor_kernels()
    stride = tile_size

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            if (y_end - y) < tile_size // 2 or (x_end - x) < tile_size // 2:
                continue

            rgb_tile = rgb[y:y_end, x:x_end]
            gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
            gray_norm = gray_tile.astype(np.float32) / 255.0

            stack = np.stack([cv2.filter2D(gray_norm, cv2.CV_32F, k) for k in kernels], -1)
            max_resp = np.max(stack, -1)
            if max_resp.max() > max_resp.min():
                max_resp = (max_resp - max_resp.min()) / (max_resp.max() - max_resp.min())
            mask = (max_resp > np.percentile(max_resp, percentile)).astype(np.uint8)

            overlay = rgb_tile.copy()
            overlay[mask == 1] = [255, 0, 0]

            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(gray_tile, cmap="gray"); ax[0].set_title("Gray"); ax[0].axis("off")
            ax[1].imshow(mask, cmap="gray");      ax[1].set_title("Mask"); ax[1].axis("off")
            ax[2].imshow(overlay);                  ax[2].set_title("Overlay"); ax[2].axis("off")
            plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------
# CLI entry point 
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Gabor fault‑structure masking (dark‑ridge polarity)"
    )
    parser.add_argument("image", help="Path to RGB image")
    parser.add_argument("--debug", action="store_true", help="visualise every tile")
    args = parser.parse_args()

    if args.debug:
        debug_tiles(args.image)
    else:
        mask, orient, _ = run(args.image)
        print("Finished. Mask shape:", mask.shape)




__all__ = ["create_gabor_kernels", "run", "debug_tiles"]

# ----------------------------------------------------------------------------
# Gabor kernel setup (psi = π → dark‑on‑bright polarity)
# ----------------------------------------------------------------------------

def create_gabor_kernels() -> tuple[list[np.ndarray], np.ndarray]:
    """Return a list of Gabor kernels (psi = π) plus their orientations (deg)."""

    kernels: list[np.ndarray] = []
    orientations = np.arange(0, 180, 16)
    for theta_deg in orientations:
        theta = np.deg2rad(theta_deg)
        kernel = cv2.getGaborKernel(
            (101, 101),
            sigma=11,
            theta=theta,
            lambd=25,
            gamma=0.2,
            psi=np.pi,  # psi = π → polarity: positive resp = dark‑on‑bright
        )
        kernels.append(kernel)
    return kernels, orientations

# ----------------------------------------------------------------------------
# Core processing
# ----------------------------------------------------------------------------

def run(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,
    percentile: int = 90,
):
    """Process an RGB image and return mask, orientation map, and magnitude."""

    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    gabor_mask = np.zeros((height, width), dtype=np.uint8)
    gabor_orientation = np.zeros((height, width), dtype=np.uint8)
    gabor_magnitude = np.zeros((height, width), dtype=np.float32)

    kernels, _ = create_gabor_kernels()
    stride = tile_size  # 0 % overlap

    # -----------------------------------------------------------------
    # Tile loop
    # -----------------------------------------------------------------
    for y in range(0, height, stride):
        for x in range(0, width, stride):
            y_end = min(y + tile_size, height)
            x_end = min(x + tile_size, width)

            # Ignore tiny edge tiles (<50 % of tile_size)
            if (y_end - y) < tile_size // 2 or (x_end - x) < tile_size // 2:
                continue

            rgb_tile = rgb[y:y_end, x:x_end]

            # Local contrast normalisation
            rgb_tile = cv2.normalize(rgb_tile, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
            gray_norm = gray_tile.astype(np.float32) / 255.0

            # Gabor filtering (signed responses retained)
            responses = [cv2.filter2D(gray_norm, cv2.CV_32F, k) for k in kernels]
            stack = np.stack(responses, axis=-1)

            max_response = np.max(stack, axis=-1)

            # Normalise within tile
            if max_response.max() > max_response.min():
                max_response = (max_response - max_response.min()) / (
                    max_response.max() - max_response.min()
                )

            thresh = np.percentile(max_response, percentile)
            tile_mask = (max_response > thresh).astype(np.uint8)
            tile_orient = np.argmax(stack, axis=-1)

            # Write back to global arrays
            gabor_mask[y:y_end, x:x_end] = np.maximum(
                gabor_mask[y:y_end, x:x_end], tile_mask
            )
            gabor_orientation[y:y_end, x:x_end] = np.where(
                tile_mask > 0, tile_orient, gabor_orientation[y:y_end, x:x_end]
            )
            gabor_magnitude[y:y_end, x:x_end] = np.maximum(
                gabor_magnitude[y:y_end, x:x_end], max_response
            )

    return gabor_mask, gabor_orientation, gabor_magnitude


def debug_tiles(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,
    percentile: int = 90,
):
    """Iterate through tiles and display (gray, mask, overlay).
    """

    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    h, w, _ = rgb.shape
    kernels, _ = create_gabor_kernels()
    stride = tile_size

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y_end, x_end = min(y + tile_size, h), min(x + tile_size, w)
            if (y_end - y) < tile_size // 2 or (x_end - x) < tile_size // 2:
                continue

            rgb_tile = rgb[y:y_end, x:x_end]
            gray_tile = cv2.cvtColor(rgb_tile, cv2.COLOR_RGB2GRAY)
            gray_norm = gray_tile.astype(np.float32) / 255.0

            stack = np.stack([cv2.filter2D(gray_norm, cv2.CV_32F, k) for k in kernels], -1)
            max_resp = np.max(stack, -1)
            if max_resp.max() > max_resp.min():
                max_resp = (max_resp - max_resp.min()) / (max_resp.max() - max_resp.min())
            mask = (max_resp > np.percentile(max_resp, percentile)).astype(np.uint8)

            overlay = rgb_tile.copy()
            overlay[mask == 1] = [255, 0, 0]

            # Plot
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(gray_tile, cmap="gray"); ax[0].set_title("Gray"); ax[0].axis("off")
            ax[1].imshow(mask, cmap="gray");      ax[1].set_title("Mask"); ax[1].axis("off")
            ax[2].imshow(overlay);                  ax[2].set_title("Overlay"); ax[2].axis("off")
            plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------------
# CLI entry point 
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        
    )
    parser.add_argument("image", help="Path to RGB image")
    parser.add_argument("--debug", action="store_true", help="visualise every tile")
    args = parser.parse_args()

    if args.debug:
        debug_tiles(args.image)
    else:
        mask, orient, _ = run(args.image)
        print("Finished. Mask shape:", mask.shape)
