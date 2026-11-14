#!/usr/bin/env python3
"""Projector px <-> camera px for the scene in front of the rig, by Gray-code stripes.

Run on the host with the game stopped and the rig still (python with OpenCV, e.g.
../project_map_defence/.venv/bin/python). Writes out/calib/graycode.npz:
  ambient, white : camera frames with the projector black / white
  px, py         : per camera pixel, the projector pixel that lights it (cell centres, STEP px grid)
  conf           : per camera pixel, the weakest stripe contrast (0 where the projector does not reach)
"""
import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "kmsdrm"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pygame  # noqa: E402

W, H, STEP = 1920, 1080, 8  # the finest stripe is STEP projector px wide
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "calib")


def find_camera(hint="U20CAM"):
    for dev in sorted(os.listdir("/sys/class/video4linux")):
        with open(f"/sys/class/video4linux/{dev}/name") as f:
            if hint in f.read() and open(f"/sys/class/video4linux/{dev}/index").read().strip() == "0":
                return f"/dev/{dev}"
    sys.exit(f"no camera named *{hint}*")


def main():
    pygame.display.init()
    screen = pygame.display.set_mode((W, H), pygame.FULLSCREEN)
    cam = cv2.VideoCapture(find_camera(), cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def shot(img, settle=0.35, n=3):
        """Show a (W,H) uint8 image, return the mean of n grey camera frames."""
        surf = pygame.surfarray.make_surface(np.repeat(img[:, :, None], 3, axis=2))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        t = time.time()
        while time.time() - t < settle:
            cam.grab()
        frames = [cam.read()[1] for _ in range(n)]
        return np.mean([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames], axis=0), frames[-1]

    full = np.full((W, H), 255, np.uint8)
    for _ in range(3):  # let the exposure settle on a half-lit scene
        shot(full // 2, 0.5)
    _, ambient = shot(np.zeros((W, H), np.uint8), 0.8)
    _, white = shot(full, 0.8)
    xs, ys = np.arange(W) // STEP, np.arange(H) // STEP
    codes, conf = [], None
    for axis, cells in ((0, xs), (1, ys)):
        gray = cells ^ (cells >> 1)
        code = None
        for bit in range(int(cells.max()).bit_length() - 1, -1, -1):
            line = (((gray >> bit) & 1) * 255).astype(np.uint8)
            img = np.repeat(line[:, None], H, axis=1) if axis == 0 else np.repeat(line[None, :], W, axis=0)
            a, _ = shot(img)
            b, _ = shot(255 - img)
            code = ((a > b).astype(np.int32)) if code is None else (code << 1) | (a > b)
            contrast = np.abs(a - b)
            conf = contrast if conf is None else np.minimum(conf, contrast)
        binary = code.copy()  # Gray -> binary
        shift = code >> 1
        while shift.any():
            binary ^= shift
            shift >>= 1
        codes.append(binary * STEP + STEP // 2)
    np.savez_compressed(os.path.join(OUT, "graycode.npz"), ambient=ambient, white=white,
                        px=codes[0].astype(np.int16), py=codes[1].astype(np.int16), conf=conf.astype(np.float32))
    print("saved", OUT, "lit fraction", float((conf > 12).mean()))
    os._exit(0)


if __name__ == "__main__":
    main()
