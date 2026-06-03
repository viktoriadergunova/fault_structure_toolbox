from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .derive_odd_kernel import direction_hilbert_kernel

__all__ = ["create_gabor_kernels", "run", "compute_pc", "scale_colour_map"]


def create_gabor_kernels(
    ksize=None,
    scales=None,
    gamma=0.1,
    psi=0,
    step_deg=2,
):
    if scales is None:
        scales = [
            dict(sigma=8,   lambd=16),  # Fine geology
            dict(sigma=16,  lambd=32),  # Medium faults
            dict(sigma=32,  lambd=64),  # Major structural scarps
            dict(sigma=64,  lambd=128), # Regional tectonic trends
        ]

    if ksize is None:
        sigma_max = max(sc["sigma"] for sc in scales)
        k = int(np.ceil(6 * sigma_max))
        k = k + 1 if k % 2 == 0 else k
        ksize = (k, k)
        print(f"Auto kernel size for sigma_max={sigma_max}: {ksize}")

    kernels = []
    thetas = []
    theta_deg_list = []
    scale_ids = []

    for s_id, sc in enumerate(scales):
        sigma = sc["sigma"]
        lambd = sc["lambd"]

        for deg in np.arange(0, 180, step_deg, dtype=np.int32):
            theta = float(np.deg2rad(deg))

            k = cv2.getGaborKernel(
                ksize,
                sigma=float(sigma),
                theta=theta,
                lambd=float(lambd),
                gamma=float(gamma),
                psi=float(psi),
            ).astype(np.float32)

            k -= k.mean()
            k /= (np.sqrt((k * k).sum()) + 1e-8)

            kernels.append(k)
            thetas.append(theta)
            theta_deg_list.append(int(deg))
            scale_ids.append(int(s_id))

    return (
        kernels,
        np.array(thetas,         dtype=np.float32),
        np.array(theta_deg_list, dtype=np.int32),
        np.array(scale_ids,      dtype=np.int32),
        scales,
    )


def run(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,
    do_clahe: bool = False,
    return_eo: bool = False,
    return_kernels: bool = False,
    ksize=None,
    scales=None,
    gamma: float = 0.2,
    psi: float = 0.0,
    step_deg: int = 2,
):
    # ── Load + gray ───────────────────────────────────────────────────────────
    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray0 = clahe.apply(
            (gray0 * 255).astype(np.uint8)
        ).astype(np.float32) / 255.0

    # ── Build filter bank ─────────────────────────────────────────────────────
    even_kernels, thetas_raw, theta_deg_raw, scale_ids_raw, scales_used = \
        create_gabor_kernels(
            ksize=ksize, scales=scales, gamma=gamma,
            psi=psi, step_deg=step_deg,
        )

    unique_theta_deg = np.unique(theta_deg_raw)
    deg_to_idx  = {int(d): i for i, d in enumerate(unique_theta_deg)}
    theta_idx   = np.array(
        [deg_to_idx[int(d)] for d in theta_deg_raw], dtype=np.int32
    )
    unique_thetas = np.deg2rad(unique_theta_deg.astype(np.float32))

    n_scales = int(scale_ids_raw.max()) + 1
    n_theta  = int(unique_theta_deg.shape[0])

    # ── Build odd kernels ─────────────────────────────────────────────────────
    odd_kernels = [
        direction_hilbert_kernel(k, float(th))
        for k, th in zip(even_kernels, thetas_raw)
    ]

    # ── Global pad ────────────────────────────────────────────────────────────
    pad  = even_kernels[0].shape[0] // 2
    gray = cv2.copyMakeBorder(
        gray0, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT
    )
    Hp, Wp = gray.shape

    # ── Allocate outputs ──────────────────────────────────────────────────────
    AMP = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
    PHI = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)

    if return_eo:
        E = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
        O = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
    else:
        E = O = None

    # ── Tile loop ─────────────────────────────────────────────────────────────
    stride = tile_size
    for y in range(0, Hp, stride):
        for x in range(0, Wp, stride):
            y2 = min(y + tile_size, Hp)
            x2 = min(x + tile_size, Wp)

            if (y2 - y) < tile_size // 2 or (x2 - x) < tile_size // 2:
                continue

            y0 = max(0, y - pad);   x0 = max(0, x - pad)
            y3 = min(Hp, y2 + pad); x3 = min(Wp, x2 + pad)

            tile_halo = gray[y0:y3, x0:x3]
            iy0, ix0  = y - y0, x - x0
            iy1, ix1  = iy0 + (y2 - y), ix0 + (x2 - x)

            for i in range(len(even_kernels)):
                s = int(scale_ids_raw[i])
                t = int(theta_idx[i])

                e_full = cv2.filter2D(tile_halo, cv2.CV_32F,
                                      even_kernels[i],
                                      borderType=cv2.BORDER_CONSTANT)
                o_full = cv2.filter2D(tile_halo, cv2.CV_32F,
                                      odd_kernels[i],
                                      borderType=cv2.BORDER_CONSTANT)

                e = e_full[iy0:iy1, ix0:ix1]
                o = o_full[iy0:iy1, ix0:ix1]

                AMP[s, t, y:y2, x:x2] = np.sqrt(e * e + o * o)
                PHI[s, t, y:y2, x:x2] = np.arctan2(o, e)

                if return_eo:
                    E[s, t, y:y2, x:x2] = e
                    O[s, t, y:y2, x:x2] = o

    # ── Unpad ─────────────────────────────────────────────────────────────────
    AMP = AMP[:, :, pad:-pad, pad:-pad]
    PHI = PHI[:, :, pad:-pad, pad:-pad]

    if return_eo:
        E = E[:, :, pad:-pad, pad:-pad]
        O = O[:, :, pad:-pad, pad:-pad]

    out = {
        "AMP":       AMP,
        "PHI":       PHI,
        "thetas":    unique_thetas,
        "theta_deg": unique_theta_deg,
        "scales":    scales_used,
        "scale_ids": np.arange(n_scales, dtype=np.int32),
    }

    if return_eo:
        out["E"] = E
        out["O"] = O

    if return_kernels:
        out["kernels"] = {
            "even":             even_kernels,
            "odd":              odd_kernels,
            "theta_deg_raw":    theta_deg_raw,
            "theta_raw":        thetas_raw,
            "theta_idx":        theta_idx,
            "scale_id":         scale_ids_raw,
            "theta_deg_unique": unique_theta_deg,
            "thetas_unique":    unique_thetas,
            "scales":           scales_used,
        }

    return out


def compute_pc(AMP, PHI, k=3.0, q=0.5, eps=1e-6):
    """
    Phase congruency with Jacobian-corrected AMP² weighting.

    AMP² weights — Jacobian of the polar transform (E,O)→(AMP,φ) is AMP,
    so the correct likelihood weight is AMP². Noise in AMP² space follows
    an exponential distribution — no Gaussian 1.4826 factor.
    """
    AMP2 = AMP ** 2                                        # (n_scales, n_theta, H, W)

    C         = np.sum(AMP2 * np.cos(PHI), axis=0)
    S         = np.sum(AMP2 * np.sin(PHI), axis=0)
    phi_mean  = np.arctan2(S, C)
    delta_phi = PHI - phi_mean[None, :, :, :]
    spread    = np.cos(delta_phi) - np.abs(np.sin(delta_phi))
    R         = np.sum(AMP2 * spread, axis=0)              # (n_theta, H, W)
    Asum      = np.sum(AMP2, axis=0) + eps

    n_theta = AMP.shape[1]
    PC_t    = np.zeros_like(R)

    for t in range(n_theta):
        Rt            = R[t]
        r_q           = np.quantile(Rt, q)
        noise_samples = Rt[Rt <= r_q]
        noise_median  = np.median(noise_samples)
        mad           = np.median(np.abs(noise_samples - noise_median)) + eps
        T_floor       = noise_median + k * mad
        PC_t[t]       = np.maximum(Rt - T_floor, 0.0) / Asum[t]

    PC_max         = np.max(PC_t, axis=0)
    best_theta_idx = np.argmax(PC_t, axis=0)

    return PC_max, PC_t, best_theta_idx


def scale_colour_map(AMP, PC_max, scale_colours=None, eps=1e-6):
    """
    Assign each pixel a colour based on its dominant scale.

    Winner-takes-all: the scale with the strongest AMP² response
    at each pixel determines the hue. Brightness is PC strength.

    Parameters
    ----------
    AMP          : (n_scales, n_theta, H, W)  — from run()
    PC_max       : (H, W)                     — from compute_pc()
    scale_colours: list of (R,G,B) tuples in [0,1], one per scale.
                   Default: red=fine … blue=coarse
    eps          : numerical stability

    Returns
    -------
    rgb : (H, W, 3) float32 in [0, 1]
    """
    n_scales = AMP.shape[0]

    if scale_colours is None:
        # red → orange → green → blue  (fine → coarse)
        defaults = [
            (0.90, 0.10, 0.10),   # σ smallest — red
            (0.90, 0.55, 0.00),   # orange
            (0.10, 0.75, 0.10),   # green
            (0.10, 0.30, 0.90),   # blue
        ]
        scale_colours = defaults[:n_scales]

    scale_colours = np.array(scale_colours, dtype=np.float32)  # (n_scales, 3)

    # Dominant scale per pixel — argmax of AMP² summed over orientations
    AMP2_per_scale = (AMP ** 2).max(axis=1)          # (n_scales, H, W)
    dominant_scale = np.argmax(AMP2_per_scale, axis=0)  # (H, W)

    # Build colour image from winner scale
    H, W = dominant_scale.shape
    rgb  = scale_colours[dominant_scale.ravel()].reshape(H, W, 3)

    # Brightness = normalised PC_max
    pc_norm = PC_max / (PC_max.max() + eps)          # (H, W)
    rgb     = rgb * pc_norm[:, :, np.newaxis]

    return rgb.astype(np.float32)