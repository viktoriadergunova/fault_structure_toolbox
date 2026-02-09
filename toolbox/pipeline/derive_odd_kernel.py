import numpy as np

def fftfreq(h, w):
    fy = np.fft.fftfreq(h)  # cycles/pixel
    fx = np.fft.fftfreq(w)
    kx, ky = np.meshgrid(fx, fy)
    return kx, ky

# testen ob odd kernel stabil ist 

def direction_hilbert_kernel(even_kernel: np.ndarray, theta: float,
                                    eps: float = 0.02,
                                    smooth: bool = True) -> np.ndarray:
    """
    Build odd kernel from even kernel via directional Hilbert transform along theta.
    Stabilized by:
      - consistent centering (ifftshift/fftshift)
      - smooth/robust sign mask in Fourier domain
    eps: softness / deadzone parameter in frequency units (cycles/pixel)
    """

    kh, kw = even_kernel.shape

    # --- Center kernel before FFT (critical) ---
    ke = np.fft.ifftshift(even_kernel.astype(np.float32))
    Ge = np.fft.fft2(ke)

    # --- Directional mask ---
    kx, ky = fftfreq(kh, kw)
    nx, ny = np.cos(theta), np.sin(theta)
    proj = kx * nx + ky * ny  # signed projection

    if smooth:
        # smooth sign: continuous, reduces ringing
        sgn = np.tanh(proj / (eps + 1e-12)).astype(np.float32)
    else:
        # hard sign with deadzone
        sgn = np.sign(proj).astype(np.float32)
        sgn[np.abs(proj) < eps] = 0.0

    sgn[0, 0] = 0.0  # DC defined as 0

    Htheta = (-1j * sgn)  # directional Hilbert multiplier
    Go = Htheta * Ge

    odd = np.fft.ifft2(Go).real
    odd = np.fft.fftshift(odd)  # back to centered kernel coordinates

    # --- Normalize ---
    odd -= odd.mean()
    odd /= (np.sqrt((odd * odd).sum()) + 1e-8)

    return odd.astype(np.float32)


# Orientierung und Kernelgröße müssen übereinstimmen!