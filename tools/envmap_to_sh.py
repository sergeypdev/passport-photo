"""Project an equirectangular HDR environment map onto DPR's order-2 SH lighting.

DPR's SH_basis already folds in the Lambertian irradiance factors (att = pi*[1,2/3,1/4]),
and it reconstructs shading as sum_i sh_i * SH_basis_i(normal). So the DPR-compatible
coefficients are just the plain (orthonormal, no-att) real-SH projection of the env
luminance:  sh_i = integral L(w) * Y_i(w) dOmega.

Axis convention (verified against DPR's own shading renderer):
  x = right,  z = up (cos theta),  -y = toward camera (front).

Usage: python envmap_to_sh.py /path/to/env.exr
"""
import os
import sys

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "DPR", "utils"))
from utils_SH import get_shading  # noqa: E402

TARGET_DC = 0.75  # match DPR's demo SH scale (example DC ~1.08 * 0.7)


def real_sh9(x, y, z):
    """DPR's no-att (orthonormal real) SH basis, coeff order [DC, y, z, x, ...]."""
    Y = np.empty((9,) + x.shape)
    Y[0] = 0.5 / np.sqrt(np.pi) * np.ones_like(x)
    c1 = np.sqrt(3) / 2 / np.sqrt(np.pi)
    Y[1] = c1 * y
    Y[2] = c1 * z
    Y[3] = c1 * x
    c2a = np.sqrt(15) / 2 / np.sqrt(np.pi)
    c2b = np.sqrt(5) / 4 / np.sqrt(np.pi)
    c2c = np.sqrt(15) / 4 / np.sqrt(np.pi)
    Y[4] = c2a * x * y
    Y[5] = c2a * y * z
    Y[6] = c2b * (3 * z * z - 1)
    Y[7] = c2a * x * z
    Y[8] = c2c * (x * x - y * y)
    return Y


def main(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"failed to read {path}")
    img = np.nan_to_num(img.astype(np.float64), posinf=0, neginf=0)
    lum = 0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]  # BGR
    H, W = lum.shape

    theta = (np.arange(H) + 0.5) / H * np.pi        # 0 top .. pi bottom
    phi = (np.arange(W) + 0.5) / W * 2 * np.pi
    TH, PH = np.meshgrid(theta, phi, indexing="ij")
    sinT = np.sin(TH)
    domega = sinT * (np.pi / H) * (2 * np.pi / W)

    def project(rot):
        ang = PH + rot
        z = np.cos(TH)
        x = sinT * np.sin(ang)
        y = -sinT * np.cos(ang)  # -y is front, so ang=0 -> front
        Y = real_sh9(x, y, z)
        return np.array([np.sum(lum * Y[i] * domega) for i in range(9)])

    # Auto-orient: rotate the map about vertical so the key light faces the camera
    # (align the horizontal l=1 vector (x=coeff3, front=-coeff1) to front).
    c0 = project(0.0)
    rot = -np.arctan2(c0[3], -c0[1])
    c = project(rot)
    c = c / c[0] * TARGET_DC  # normalise DC (keeps directional ratios)

    np.set_printoptions(precision=4, suppress=True)
    print("auto-rotation (deg):", round(float(np.degrees(rot)), 1))
    print("SH:", [round(float(v), 4) for v in c])

    # Diagnostics: render DPR shading on its image-space normal hemisphere.
    N = 256
    xx = np.linspace(-1, 1, N)
    zz = np.linspace(1, -1, N)
    xx, zz = np.meshgrid(xx, zz)
    valid = np.sqrt(xx**2 + zz**2) <= 1
    yy = -np.sqrt(np.clip(1 - (xx * valid) ** 2 - (zz * valid) ** 2, 0, 1))
    normal = np.stack([xx * valid, yy * valid, zz * valid], -1).reshape(-1, 3)
    s = get_shading(normal, c.astype(np.float32)).reshape(N, N)
    s = np.where(valid, s, np.nan)
    r, col = np.unravel_index(np.nanargmax(s), s.shape)
    print(f"shading min/max = {np.nanmin(s):.3f}/{np.nanmax(s):.3f}")
    print(f"brightest @ row {r}, col {col} (center ~ {N // 2})")
    print(f"L-R balance = {np.nanmean(s[:, :N//2]) - np.nanmean(s[:, N//2:]):+.3f}")
    print(f"T-B balance = {np.nanmean(s[:N//2, :]) - np.nanmean(s[N//2:, :]):+.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
