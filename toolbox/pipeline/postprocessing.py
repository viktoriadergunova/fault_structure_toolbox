"""
Post–processing for the Gabor mask.


    postprocess(mask: np.ndarray, rgb: np.ndarray,
                tile_size: int = 512, stride: int | None = None,
                debug: bool = False) -> list[dict]

* **mask** – binary mask `)
* **rgb**  – original RGB image `
* Returns a list of dictionaries, one per fitted branch segment, with keys
  `nodes`, `slope`, `intercept` (global‑image coordinates).

"""

from __future__ import annotations

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from skimage.measure import label, regionprops
from skimage.morphology import medial_axis
from scipy.stats import linregress

__all__ = ["postprocess"]


def _process_tile(
    tile_mask: np.ndarray,
    tile_rgb: np.ndarray,
    row_start: int,
    col_start: int,
    debug: bool,
):
    """Process one tile; return list of fitted line dicts and optional plots."""

    fitted = []

    if np.sum(tile_mask) == 0:
        return fitted  # nothing to do

    skeleton, _ = medial_axis(tile_mask.astype(bool), return_distance=True)
    skeleton = skeleton.astype(np.uint8)

    # Build 8‑connected graph of skeleton pixels
    rows, cols = np.where(skeleton)
    G = nx.Graph()
    for r, c in zip(rows, cols):
        G.add_node((r, c))
    for r, c in G.nodes():
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1] and skeleton[nr, nc]:
                    G.add_edge((r, c), (nr, nc))

    junctions = [n for n, d in G.degree() if d > 2]
    branches_G = G.copy()
    branches_G.remove_nodes_from(junctions)

    skel_decomp = np.zeros_like(skeleton, dtype=np.uint8)
    overlay = tile_rgb.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    for component in nx.connected_components(branches_G):
        coords = np.array(list(component))
        if coords.shape[0] < 3:
            continue

        branch_mask = np.zeros_like(skeleton, dtype=bool)
        branch_mask[tuple(coords.T)] = True
        labeled = label(branch_mask)
        props = regionprops(labeled)
        if not props:
            continue

        region = props[0]
        coords = region.coords
        x = coords[:, 1]
        y = coords[:, 0]

        if len(x) < 10:
            continue

        if np.all(x == x[0]):  # vertical line
            slope = np.inf
            intercept = x[0]
        else:
            slope, intercept, *_ = linregress(x, y)

        # accumulate points into decomp mask for debug plot
        skel_decomp[y, x] = 1

        # draw on local overlay for debug
        x_min, x_max = x.min(), x.max()
        if slope == np.inf:
            y_min, y_max = y.min(), y.max()
            pt1, pt2 = (int(x_min), int(y_min)), (int(x_max), int(y_max))
        else:
            y1, y2 = slope * x_min + intercept, slope * x_max + intercept
            pt1, pt2 = (int(x_min), int(y1)), (int(x_max), int(y2))
        cv2.line(overlay, pt1, pt2, (0, 0, 0), 3)

        # translate branch coords to global image indices
        global_nodes = [(int(y_i + row_start), int(x_i + col_start)) for x_i, y_i in zip(x, y)]
        fitted.append({"nodes": global_nodes, "slope": slope, "intercept": intercept})

    if debug:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        axs[0, 0].imshow(tile_mask, cmap="gray"); axs[0, 0].set_title("a) Mask Tile"); axs[0,0].axis("off")
        axs[0, 1].imshow(skeleton, cmap="gray"); axs[0, 1].set_title("b) Skeleton"); axs[0,1].axis("off")
        axs[1, 0].imshow(skel_decomp, cmap="gray"); axs[1, 0].set_title("c) Decomposed"); axs[1,0].axis("off")
        axs[1, 1].imshow(overlay); axs[1, 1].set_title("d) Fitted Lines"); axs[1,1].axis("off")
        plt.tight_layout(); plt.show()

    return fitted


def postprocess(
    mask: np.ndarray,
    rgb: np.ndarray,
    tile_size: int = 512,
    stride: int | None = None,
    debug: bool = False,
):
    """Skeletonise mask per tile, fit straight lines, return list of segments."""

    if stride is None:
        stride = tile_size  # no overlap

    height, width = mask.shape
    fitted_lines_data: list[dict] = []

    for row_start in range(0, height - tile_size + 1, stride):
        for col_start in range(0, width - tile_size + 1, stride):
            row_end, col_end = row_start + tile_size, col_start + tile_size
            tile_mask = mask[row_start:row_end, col_start:col_end]
            tile_rgb = rgb[row_start:row_end, col_start:col_end]

            fitted_lines_data.extend(
                _process_tile(tile_mask, tile_rgb, row_start, col_start, debug)
            )

    return fitted_lines_data
