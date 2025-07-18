"""
evaluation.py – for orientation analysis

Usage
-----
```python
from toolbox.pipeline.evaluation import evaluate_mask

segments = postprocess(mask, rgb)

evaluate_mask(mask, orient,
              rgb=rgb,
              segments=segments,
              pdf_out="tile_roses_with_segments.pdf",
              tile_size=512)
```
This prints global stats and, if `pdf_out` is provided, stores a multi‑page
PDF: each page shows the **tile image with fitted segment dots** on the left and
a rose diagram of the tile’s orientations on the right, plus a final global
rose.
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ["evaluate_mask"]

# -----------------------------------------------------------------------------
# Helper – rose plot
# -----------------------------------------------------------------------------

def _rose(ax, angles_deg: np.ndarray, title: str, bins: int):
    if angles_deg.size == 0:
        ax.set_title(title + "\n(no data)"); ax.set_axis_off(); return
    edges = np.linspace(0, 180, bins + 1)
    hist, _ = np.histogram(angles_deg, edges)
    theta = np.radians((edges[:-1] + edges[1:]) / 2)
    ax.bar(theta, hist, width=np.radians(180 / bins), color="gray", edgecolor="black")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150]))
    ax.set_xticklabels(["N(0°)", "30°", "60°", "E(90°)", "120°", "150°"])
    ax.set_title(title)

# -----------------------------------------------------------------------------
# Main evaluation function
# -----------------------------------------------------------------------------

def evaluate_mask(
    mask: np.ndarray,
    orientation: np.ndarray,
    rgb: np.ndarray | None = None,
    segments: list[dict] | None = None,
    pdf_out: str | Path | None = None,
    tile_size: int = 512,
    bins: int = 36,
):
    """Create per‑tile rose diagrams with fitted‑segment overlay.

    Parameters
    ----------
    mask, orientation : np.ndarray
        From `gabor.run()`, same shape.
    rgb : np.ndarray, optional
        RGB image (same crop/size). Required for overlay.
    segments : list[dict], optional
        Output of `postprocess()`; each dict must have key "nodes" holding
        `(row, col)` tuples.
    pdf_out : path or None
        If provided, save multi‑page PDF.
    tile_size : int
        Tile edge length (match detection step).
    bins : int
        Number of 5‑degree bins (36 ⇒ 5°).
    """

    assert mask.shape == orientation.shape, "mask & orientation size mismatch"
    h, w = mask.shape
    if rgb is not None:
        assert rgb.shape[:2] == mask.shape, "rgb must match mask size"

    pdf = PdfPages(str(pdf_out)) if pdf_out else None
    all_angles: list[float] = []

    tid = 0
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tid += 1
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)
            m = mask[y:y2, x:x2]
            o = orientation[y:y2, x:x2]
            angles = (o[m == 1] * 16).astype(float)
            all_angles.extend(angles)

            if pdf is None:
                continue  # no per‑tile plotting unless PDF requested

            if rgb is None:
                raise ValueError("rgb image required for plotting")

            img_tile = rgb[y:y2, x:x2].copy()
            # overlay fitted segments if given
            if segments:
                for seg in segments:
                    for (r, c) in seg["nodes"]:
                        if y <= r < y2 and x <= c < x2:
                            rr, cc = r - y, c - x
                            cv2.circle(img_tile, (cc, rr), 1, (0, 0, 0), -1)

            fig = plt.figure(figsize=(8, 4))
            ax_img = fig.add_subplot(1, 2, 1)
            ax_rose = fig.add_subplot(1, 2, 2, projection="polar")
            ax_img.imshow(img_tile); ax_img.axis("off"); ax_img.set_title(f"Tile {tid}")
            _rose(ax_rose, angles, "Orientation", bins)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    all_angles = np.asarray(all_angles)
    print("Mask pixels analysed:", all_angles.size)
    if all_angles.size:
        mean_ang = np.degrees(np.arctan2(np.mean(np.sin(np.radians(all_angles))), np.mean(np.cos(np.radians(all_angles))))) % 180
        print(f"Circular mean orientation: {mean_ang:.2f}°")
        if pdf:
            fig = plt.figure(figsize=(5, 5)); ax = fig.add_subplot(111, projection="polar")
            _rose(ax, all_angles, "Global", bins)
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

    if pdf:
        pdf.close(); print("PDF saved to", pdf_out)
