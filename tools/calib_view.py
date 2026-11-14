#!/usr/bin/env python3
"""Side-by-side check of a projector model: the camera photo in projector pixels with the
lidar's depth edges drawn on it, and the lidar depth image. Usage:
  calib_view.py out.jpg throw offx offy x y z roll pitch yaw [step]"""
import sys

import cv2
import numpy as np

import calib_lib as L


def edges(depth):
    d = np.where(np.isfinite(depth), depth, 30.0)
    e = np.zeros(d.shape, bool)
    e[:, 1:] |= (d[:, :-1] - d[:, 1:]) > np.maximum(0.12, 0.04 * d[:, 1:])   # the nearer side of each jump
    e[:, :-1] |= (d[:, 1:] - d[:, :-1]) > np.maximum(0.12, 0.04 * d[:, :-1])
    e[1:, :] |= (d[:-1, :] - d[1:, :]) > np.maximum(0.12, 0.04 * d[1:, :])
    e[:-1, :] |= (d[1:, :] - d[:-1, :]) > np.maximum(0.12, 0.04 * d[:-1, :])
    return e


def main():
    out, a = sys.argv[1], list(map(float, sys.argv[2:11]))
    gc = np.load("out/calib/graycode.npz")
    pts = np.load("out/calib/scans.npz")["points"][:, :3].astype(float)
    photo, _ = L.photo_dense(gc)
    photo = cv2.resize(photo, (L.W // 8, L.H // 8), interpolation=cv2.INTER_AREA) if False else photo
    depth = L.depth_image(pts, a[0], a[1:3], a[3:6], a[6:9])
    print("cells with lidar:", np.isfinite(depth).mean().round(3), "depth pct", np.percentile(depth[np.isfinite(depth)], [5, 50, 95]).round(2))
    k = 5
    big = cv2.resize(photo, (depth.shape[1] * k, depth.shape[0] * k), interpolation=cv2.INTER_LINEAR)
    e = cv2.resize(edges(depth).astype(np.uint8), None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST) > 0
    big[e] = (0, 255, 0)
    dn = np.clip((depth - 0.5) / 5.5, 0, 1)
    col = cv2.applyColorMap((255 - dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[~np.isfinite(depth)] = 0
    col = cv2.resize(col, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(out, np.vstack([big, col]))


main()
