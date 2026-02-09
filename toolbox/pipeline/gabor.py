from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

from .derive_odd_kernel import direction_hilbert_kernel

__all__ = ["create_gabor_kernels", "run"]


def create_gabor_kernels(
    ksize=None,  # Jetzt optional!
    scales=None,
    gamma=0.1,
    psi=0,
    step_deg=16,
):
    """
    Returns:
      kernels:  list[np.ndarray]      (n_kernels,) each (ksize, ksize) float32, L2-normalized
      thetas:   np.ndarray float32    (n_kernels,) theta in radians
      theta_deg:np.ndarray int32      (n_kernels,) theta in degrees (0..179)
      scale_ids:np.ndarray int32      (n_kernels,) scale index
      scales:   list[dict]            list of {"sigma":..., "lambd":...}
    """
    if scales is None:
        scales = [
            dict(sigma=3,   lambd=6),    # feinste Details
            dict(sigma=6,   lambd=12),   # deine erste
            dict(sigma=12,  lambd=24),   # mittlere Strukturen
            dict(sigma=24,  lambd=48),   # grobe Strukturen
        ]

    # Adaptive Kernel-Größe basierend auf größtem sigma
    if ksize is None:
        sigma_max = max(sc["sigma"] for sc in scales)
        k = int(np.ceil(6 * sigma_max))  # Faustregel: 6*sigma
        k = k + 1 if k % 2 == 0 else k   # Ungerade machen
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
        np.array(thetas, dtype=np.float32),
        np.array(theta_deg_list, dtype=np.int32),
        np.array(scale_ids, dtype=np.int32),
        scales,
    )


def run(
    image_path: str | Path,
    tile_size: int = 512,
    crop: int = 0,
    do_clahe: bool = False,
    return_eo: bool = False,
    return_kernels: bool = False,
    ksize=None,  # Jetzt optional!
    scales=None,
    gamma: float = 0.2,
    psi: float = 0.0,
    step_deg: int = 16,
):
    """
    Quadrature Gabor bank (even + odd via directional Hilbert), tiled with HALO.

    Returns dict:
      AMP: (S, T, H, W) float32
      PHI: (S, T, H, W) float32
      thetas: (T,) float32 radians
      theta_deg: (T,) int32 degrees
      scales: list of dicts (len S)
      scale_ids: (S,) int32 (0..S-1)
      optional:
        E/O: (S, T, H, W) float32
        kernels: dict with even/odd and mappings
    """
    # --------------------------
    # Load + gray
    # --------------------------
    rgb = cv2.imread(str(image_path))
    if rgb is None:
        raise FileNotFoundError(image_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    if crop:
        rgb = rgb[crop:-crop, crop:-crop]

    gray0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    if do_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray0 = clahe.apply((gray0 * 255).astype(np.uint8)).astype(np.float32) / 255.0

    # --------------------------
    # Build filterbank (even kernels + theta_deg for stable indexing)
    # --------------------------
    even_kernels, thetas_raw, theta_deg_raw, scale_ids_raw, scales_used = create_gabor_kernels(
        ksize=ksize,  # Wird automatisch berechnet wenn None
        scales=scales,
        gamma=gamma,
        psi=psi,
        step_deg=step_deg,
    )

    # Stable theta indexing using integer degrees
    unique_theta_deg = np.unique(theta_deg_raw)  # (T,)
    deg_to_idx = {int(d): i for i, d in enumerate(unique_theta_deg)}
    theta_idx = np.array([deg_to_idx[int(d)] for d in theta_deg_raw], dtype=np.int32)

    unique_thetas = np.deg2rad(unique_theta_deg.astype(np.float32))  # (T,) radians

    n_scales = int(scale_ids_raw.max()) + 1
    n_theta = int(unique_theta_deg.shape[0])

    # --------------------------
    # Build odd kernels ONCE
    # --------------------------
    odd_kernels = [
        direction_hilbert_kernel(k, float(th))
        for k, th in zip(even_kernels, thetas_raw)
    ]

    # --------------------------
    # Global pad once (reflect)
    # --------------------------
    pad = even_kernels[0].shape[0] // 2
    gray = cv2.copyMakeBorder(gray0, pad, pad, pad, pad, borderType=cv2.BORDER_REFLECT)
    Hp, Wp = gray.shape

    # --------------------------
    # Allocate PADDED outputs
    # --------------------------
    AMP = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
    PHI = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)

    if return_eo:
        E = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
        O = np.zeros((n_scales, n_theta, Hp, Wp), dtype=np.float32)
    else:
        E = O = None

    # --------------------------
    # Tile loop with HALO
    # --------------------------
    stride = tile_size
    for y in range(0, Hp, stride):
        for x in range(0, Wp, stride):
            y2 = min(y + tile_size, Hp)
            x2 = min(x + tile_size, Wp)

            if (y2 - y) < tile_size // 2 or (x2 - x) < tile_size // 2:
                continue

            y0 = max(0, y - pad)
            x0 = max(0, x - pad)
            y3 = min(Hp, y2 + pad)
            x3 = min(Wp, x2 + pad)

            tile_halo = gray[y0:y3, x0:x3]

            iy0 = y - y0
            ix0 = x - x0
            iy1 = iy0 + (y2 - y)
            ix1 = ix0 + (x2 - x)

            for i in range(len(even_kernels)):
                s = int(scale_ids_raw[i])
                t = int(theta_idx[i])

                e_full = cv2.filter2D(tile_halo, cv2.CV_32F, even_kernels[i], borderType=cv2.BORDER_CONSTANT)
                o_full = cv2.filter2D(tile_halo, cv2.CV_32F, odd_kernels[i],  borderType=cv2.BORDER_CONSTANT)

                e = e_full[iy0:iy1, ix0:ix1]
                o = o_full[iy0:iy1, ix0:ix1]

                AMP[s, t, y:y2, x:x2] = np.sqrt(e * e + o * o)
                PHI[s, t, y:y2, x:x2] = np.arctan2(o, e)

                if return_eo:
                    E[s, t, y:y2, x:x2] = e
                    O[s, t, y:y2, x:x2] = o

    # --------------------------
    # UNPAD back to original size
    # --------------------------
    AMP = AMP[:, :, pad:-pad, pad:-pad]
    PHI = PHI[:, :, pad:-pad, pad:-pad]

    if return_eo:
        E = E[:, :, pad:-pad, pad:-pad]
        O = O[:, :, pad:-pad, pad:-pad]

    out = {
        "AMP": AMP,
        "PHI": PHI,
        "thetas": unique_thetas,
        "theta_deg": unique_theta_deg,
        "scales": scales_used,
        "scale_ids": np.arange(n_scales, dtype=np.int32),
    }

    if return_eo:
        out["E"] = E
        out["O"] = O

    if return_kernels:
        out["kernels"] = {
            "even": even_kernels,
            "odd": odd_kernels,
            "theta_deg_raw": theta_deg_raw,
            "theta_raw": thetas_raw,
            "theta_idx": theta_idx,
            "scale_id": scale_ids_raw,
            "theta_deg_unique": unique_theta_deg,
            "thetas_unique": unique_thetas,
            "scales": scales_used,
        }

    return out

def compute_pc(AMP, PHI, k=3.0, q=0.5, eps=1e-6):
    """
    Kovesi Phase Congruency
    
    Returns:
        PC_max: (H, W) - Maximale Kongruenz über alle Orientierungen
        PC_t:   (T, H, W) - Kongruenz pro Orientierung
        Or:     (H, W) - Dominante Orientierung (in Radiant)
    """
    # 1. Vektorsummen pro Orientierung (Symmetrie-Check)
    C = np.sum(AMP * np.cos(PHI), axis=0)  
    S = np.sum(AMP * np.sin(PHI), axis=0)  
    
    # Mittlere Phase pro Orientierung (Hier steckt die Symmetrie-Info!)
    phi_mean = np.arctan2(S, C)  
    
    # 2. Phasenabweichungen & Spread-Term
    delta_phi = PHI - phi_mean[None, :, :, :]  
    spread = np.cos(delta_phi) - np.abs(np.sin(delta_phi))
    
    # 3. Gewichtete Response & Energie
    R = np.sum(AMP * spread, axis=0)  
    Asum = np.sum(AMP, axis=0) + eps
    
    n_theta = AMP.shape[1]
    PC_t = np.zeros_like(R)
    
    # 4. Robuste Rauschschätzung pro Orientierung
    for t in range(n_theta):
        Rt = R[t]
        # Wir nehmen das q-Quantil für die Null-Linie des Rauschens
        r_q = np.quantile(Rt, q)
        noise_samples = Rt[Rt <= r_q]
        
        mad = np.median(np.abs(noise_samples - np.median(noise_samples))) + eps
        sigma = 1.4826 * mad
        T_floor = np.median(noise_samples) + k * sigma
        
        # PC berechnen: (Energie - Noise) / Amplitudensumme
        PC_t[t] = np.maximum(Rt - T_floor, 0.0) / Asum[t]

    # 5. Finale Maps aggregieren
    PC_max = np.max(PC_t, axis=0)
    best_theta_idx = np.argmax(PC_t, axis=0)
    
    return PC_max, PC_t, best_theta_idx