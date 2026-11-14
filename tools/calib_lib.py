"""Calibration helpers: the camera photo resampled into projector pixels, and the lidar cloud
rendered as a depth image through a projector model."""
import numpy as np

W, H = 1920, 1080
BASE = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # projector axes in the airy frame at rpy 0


def photo_in_projector(gc, step=8, min_conf=10):
    """Mean ambient camera colour per projector cell -> (H/step, W/step, 3) uint8, and a valid mask."""
    ok = (gc["conf"] > min_conf) & (gc["px"] < W) & (gc["py"] < H)
    cx, cy = gc["px"][ok] // step, gc["py"][ok] // step
    w, h = W // step, H // step
    idx = cy.astype(np.int64) * w + cx
    out = np.zeros((h * w, 3))
    n = np.bincount(idx, minlength=h * w)
    for c in range(3):
        out[:, c] = np.bincount(idx, weights=gc["ambient"][..., c][ok].astype(float), minlength=h * w)
    valid = n > 0
    out[valid] /= n[valid, None]
    return out.reshape(h, w, 3).astype(np.uint8), valid.reshape(h, w)


def photo_dense(gc, step=8, up=2, min_conf=10):
    """The whole ambient photo resampled into projector pixels, (H/step*up, W/step*up, 3).
    Cells the stripes could not decode (dark surfaces) take a camera position interpolated from
    their neighbours: the projector -> camera map is smooth, the two sit close together."""
    import cv2
    from scipy.interpolate import griddata
    ok = (gc["conf"] > min_conf) & (gc["px"] < W) & (gc["py"] < H)
    w, h = W // step, H // step
    idx = (gc["py"][ok] // step).astype(np.int64) * w + gc["px"][ok] // step
    n = np.bincount(idx, minlength=h * w)
    ys, xs = np.nonzero(ok)
    cam_x = np.bincount(idx, weights=xs, minlength=h * w) / np.maximum(n, 1)
    cam_y = np.bincount(idx, weights=ys, minlength=h * w) / np.maximum(n, 1)
    good = (n >= 3).reshape(h, w)
    gy, gx = np.nonzero(good)
    yy, xx = np.mgrid[0:h, 0:w]
    maps = []
    for c in (cam_x, cam_y):
        m = griddata((gy, gx), c.reshape(h, w)[good], (yy, xx), method="linear")
        near = griddata((gy, gx), c.reshape(h, w)[good], (yy, xx), method="nearest")
        m = np.where(np.isnan(m), near, m).astype(np.float32)
        maps.append(cv2.resize(cv2.medianBlur(m, 5), (w * up, h * up), interpolation=cv2.INTER_LINEAR))
    return cv2.remap(gc["ambient"], maps[0], maps[1], cv2.INTER_LINEAR), cv2.resize(good.astype(np.uint8), (w * up, h * up)) > 0


def rpy_rot(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def depth_image(pts, throw, offset, xyz, rpy_deg, step=8, splat=1, fill=2):
    """Nearest lidar depth per projector cell (inf where none). pts: (N,3) in the airy frame."""
    rot = rpy_rot(rpy_deg) @ BASE
    p = (pts - np.asarray(xyz)) @ rot
    z = p[:, 2]
    f = throw * W
    ok = z > 0.3
    u = (f * p[ok, 0] / z[ok] + W / 2 + offset[0] * W) / step
    v = (f * p[ok, 1] / z[ok] + H / 2 + offset[1] * H) / step
    z = z[ok]
    w, h = W // step, H // step
    img = np.full((h, w), np.inf)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            ui, vi = np.floor(u).astype(int) + du, np.floor(v).astype(int) + dv
            m = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
            np.minimum.at(img, (vi[m], ui[m]), z[m])
    # Close the gaps between the lidar's beam rows: empty cells take the nearer of the rows around.
    for _ in range(fill):
        up, down = np.roll(img, 1, axis=0), np.roll(img, -1, axis=0)
        up[0], down[-1] = np.inf, np.inf
        img = np.where(np.isfinite(img), img, np.minimum(up, down))
    return img
