#!/usr/bin/env python3
"""Fit the projector model to one still capture: the lidar's depth edges, seen through the
model, should fall on brightness edges of the camera photo resampled into projector pixels
(tools/capture_graycode.py + tools/dump_scans.py). Prints the best model.
Usage: calib_fit.py throw offset_y roll pitch yaw   (the starting model)"""
import sys

import cv2
import numpy as np
from scipy.optimize import minimize

import calib_lib as L

XYZ = (0.0, 0.0, -0.08)  # lens in the airy frame: by ruler, barely matters at room distances


def edge_map(photo, valid):
    g = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    solid = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    mag[~solid] = 0
    mag = cv2.GaussianBlur(mag, (0, 0), 2.0)
    return mag / mag[solid].mean(), solid


def depth_edges(depth):
    """Cells on the near side of a depth jump. The border of the lidar's coverage is no edge."""
    d = depth
    e = np.zeros(d.shape, bool)
    with np.errstate(invalid="ignore"):
        e[:, 1:] |= (d[:, :-1] - d[:, 1:]) > np.maximum(0.12, 0.04 * d[:, 1:])
        e[:, :-1] |= (d[:, 1:] - d[:, :-1]) > np.maximum(0.12, 0.04 * d[:, :-1])
        e[1:, :] |= (d[:-1, :] - d[1:, :]) > np.maximum(0.12, 0.04 * d[1:, :])
        e[:-1, :] |= (d[1:, :] - d[:-1, :]) > np.maximum(0.12, 0.04 * d[:-1, :])
    known = np.isfinite(d).astype(np.uint8)
    return e & (cv2.erode(known, np.ones((5, 5), np.uint8)) > 0)


def main():
    gc = np.load("out/calib/graycode.npz")
    pts = np.load("out/calib/scans.npz")["points"][:, :3].astype(float)
    r = np.linalg.norm(pts, axis=1)
    pts = pts[(pts[:, 0] > 0.5 * r) & (r > 0.6)][::3]  # the forward cone
    photo, valid = L.photo_dense(gc)
    size = (L.W // 8, L.H // 8)
    photo, valid = cv2.resize(photo, size, interpolation=cv2.INTER_AREA), cv2.resize(valid.astype(np.uint8), size) > 0
    mag, solid = edge_map(photo, valid)
    offy = float(sys.argv[2])

    def score(p):
        throw, roll, pitch, yaw = p
        e = depth_edges(L.depth_image(pts, throw, (0.0, offy), XYZ, (roll, pitch, yaw))) & solid
        return 0.0 if e.sum() < 100 else float(mag[e].mean())

    # Local search only: start from a model that already roughly fits (see calib_view.py). The
    # lens offset is not fitted; over the projector's narrow field it only trades against pitch.
    p0 = np.array([float(sys.argv[1]), *map(float, sys.argv[3:6])])
    best = (score(p0), p0)
    print(f"start {best[0]:.3f}")
    for scale in (1.0, 0.4):
        simplex = best[1] + np.vstack([np.zeros(4), np.diag([0.04, 1.0, 0.7, 0.7]) * scale])
        res = minimize(lambda q: -score(q), best[1], method="Nelder-Mead",
                       options={"initial_simplex": simplex, "xatol": 0.005, "fatol": 1e-4, "maxiter": 300})
        if -res.fun > best[0]:
            best = (-res.fun, res.x)
    s, p = best
    print(f"best {s:.3f} throw {p[0]:.3f} offy {offy} rpy {p[1]:.2f} {p[2]:.2f} {p[3]:.2f}")
    print("view args:", f"{p[0]:.3f} 0 {offy} {XYZ[0]} {XYZ[1]} {XYZ[2]} {p[1]:.2f} {p[2]:.2f} {p[3]:.2f}")

main()
