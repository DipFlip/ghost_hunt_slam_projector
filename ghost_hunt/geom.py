"""Shared maths: rotations and the 10 cm voxel keys."""
import numpy as np

VOXEL = 0.10  # metres; the grid of /slam/scan_preview
_OFF = 1 << 20


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rot_exp(v):
    """Rotation matrix of the rotation vector v."""
    a = float(np.linalg.norm(v))
    if a < 1e-9:
        return np.eye(3)
    k = v / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def pack(ijk):
    ijk = ijk.astype(np.int64) + _OFF
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


def unpack(keys):
    ijk = np.stack([keys >> 42, (keys >> 21) & 0x1FFFFF, keys & 0x1FFFFF], axis=1) - _OFF
    return (ijk + 0.5) * VOXEL
