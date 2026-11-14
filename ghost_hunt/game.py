"""Ghost hunt: the projector is a torch that shows ghosts fixed in the room.

Each frame draws the world from the projector's own pose with the projector's own lens model,
so a ghost on a wall lands on that wall whatever the keystone. Aim the reticle at one for
catch_seconds to catch it.
"""
import math
import os
import signal
import time

import numpy as np
import yaml

HEADLESS = os.environ.get("GHOST_HEADLESS", "0") == "1"
os.environ.setdefault("SDL_VIDEODRIVER", "dummy" if HEADLESS else "kmsdrm")
import pygame  # noqa: E402

from .geom import VOXEL, pack, unpack  # noqa: E402
from .link import Link  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "config.yaml")
# Projector axes (x right, y down, z out of the lens) in the airy frame when rpy is zero.
BANDS = np.array([(255, 60, 60), (255, 200, 0), (60, 255, 60), (0, 220, 255), (90, 90, 255), (255, 60, 255)], dtype=np.uint8)
BASE = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])


def rpy_rot(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


class View:
    """The projector as a camera: odom points -> pixels."""

    def __init__(self, cfg, size, pose):
        self.w, self.h = size
        self.f = cfg["throw_ratio"] * self.w
        self.cx = self.w / 2 + cfg["lens_offset"][0] * self.w
        self.cy = self.h / 2 + cfg["lens_offset"][1] * self.h
        pos, rot = pose
        self.rot = rot @ rpy_rot(cfg["rpy_deg"]) @ BASE  # odom <- projector
        self.pos = pos + rot @ np.array(cfg["xyz"], dtype=float)
        aim = np.array([(self.w / 2 - self.cx) / self.f, (self.h / 2 - self.cy) / self.f, 1.0])
        self.aim = aim / np.linalg.norm(aim)  # ray through the image centre, projector frame

    def project(self, pts):
        """(N,3) odom points -> pixel u, v and depth (depth <= 0: behind the lens)."""
        p = (np.atleast_2d(pts) - self.pos) @ self.rot
        z = np.where(p[:, 2] > 1e-3, p[:, 2], np.nan)
        return self.f * p[:, 0] / z + self.cx, self.f * p[:, 1] / z + self.cy, p[:, 2]

    def aim_angle_deg(self, pt):
        d = (pt - self.pos) @ self.rot
        return math.degrees(math.acos(np.clip(d @ self.aim / np.linalg.norm(d), -1, 1)))

    def frustum(self, depth=2.0):
        corners = [(0, 0), (self.w, 0), (self.w, self.h), (0, self.h)]
        rays = np.array([[(u - self.cx) / self.f, (v - self.cy) / self.f, 1.0] for u, v in corners]) * depth
        return np.vstack([self.pos, self.pos + rays @ self.rot.T])


def occluded(keys, a, b):
    """True if the map has surface on the segment a -> b, away from both ends."""
    length = np.linalg.norm(b - a)
    s = np.arange(0.4, length - 0.4, VOXEL * 0.7)
    if len(keys) == 0 or len(s) == 0:
        return False
    k = np.unique(pack(np.floor((a + np.outer(s / length, b - a)) / VOXEL)))
    i = np.searchsorted(keys, k)
    return int((keys[np.minimum(i, len(keys) - 1)] == k).sum()) >= 2


def spawn(keys, player, cfg, rng):
    """Ghost positions on mapped surfaces around the player."""
    if len(keys) < 500:
        return []
    pts = unpack(keys)
    floor = np.percentile(pts[:, 2], 2)
    dist = np.linalg.norm(pts[:, :2] - player[:2], axis=1)
    height = pts[:, 2] - floor
    ok = ((dist > cfg["spawn_range"][0]) & (dist < cfg["spawn_range"][1])
          & (height > cfg["spawn_height"][0]) & (height < cfg["spawn_height"][1]))
    picked = []
    for p in rng.permutation(pts[ok]):
        if all(np.linalg.norm(p - q) > cfg["min_separation"] for q in picked):
            picked.append(p)
            if len(picked) == cfg["ghosts"]:
                break
    # Just off the surface, towards where the player stood, so the surface itself does not hide it.
    return [p + 0.12 * (player - p) / np.linalg.norm(player - p) for p in picked]


def ghost_sprite():
    s = pygame.Surface((200, 240), pygame.SRCALPHA)
    white = (255, 255, 255, 255)
    pygame.draw.circle(s, white, (100, 100), 95)
    pygame.draw.rect(s, white, (5, 100, 190, 100))
    for x in (37, 100, 163):
        pygame.draw.circle(s, white, (x, 200), 32)
    for x in (65, 135):
        pygame.draw.ellipse(s, (0, 0, 0, 255), (x - 16, 70, 32, 46))
    pygame.draw.ellipse(s, (0, 0, 0, 255), (82, 135, 36, 40))
    return s


class Game:
    def __init__(self):
        pygame.display.init()
        pygame.font.init()
        if HEADLESS:
            self.canvas = pygame.display.set_mode((1280, 720))
        else:
            # 720p: SDL copies every frame to the screen on the CPU, and at 1080p that copy alone
            # (35 ms next to the SLAM) holds the game at 19 fps. The lens model is resolution-free.
            self.canvas = pygame.display.set_mode((1280, 720), pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        self.size = self.canvas.get_size()
        self.font = pygame.font.Font(None, self.size[1] // 12)
        self.sprite = ghost_sprite()
        self.ros = Link()
        self.rng = np.random.default_rng()
        self.cfg, self.cfg_mtime = None, 0
        self.load_config()
        self.overlay = False
        self.flip_s = 0.0
        self.enter("SCAN")
        print(f"ghost_hunt: display {self.size}, headless={HEADLESS}", flush=True)

    def load_config(self):
        try:
            mtime = os.path.getmtime(CONFIG)
            if mtime != self.cfg_mtime:
                with open(CONFIG) as f:
                    self.cfg = yaml.safe_load(f)
                self.cfg_mtime = mtime
        except Exception as e:  # a half-saved file: keep the last good config
            print(f"config.yaml not loaded: {e}", flush=True)

    def enter(self, phase):
        self.phase, self.phase_t = phase, time.monotonic()
        if phase == "SCAN":
            self.ghosts = []

    def start_hunt(self, player):
        pts = spawn(self.ros.map(), player, self.cfg["game"], self.rng)
        if not pts:
            return self.enter("SCAN")
        self.ghosts = [{"pos": p, "progress": 0.0, "caught": None, "aimed": False, "seen": False} for p in pts]
        self.enter("HUNT")

    def command(self, cmd, player):
        if cmd == "restart" and player is not None:
            self.start_hunt(player)
        elif cmd == "rescan":
            self.ros.clear_map()
            self.enter("SCAN")
        elif cmd == "map":
            self.overlay = not self.overlay
        elif cmd == "snap":  # what is on the projector right now, for comparing with a photo
            pygame.image.save(self.canvas, os.path.join(ROOT, "out", "frame.png"))

    def update(self, view, dt):
        g, now = self.cfg["game"], time.monotonic()
        if self.phase == "SCAN" and now - self.phase_t > g["scan_seconds"]:
            self.start_hunt(view.pos)
        elif self.phase == "HUNT":
            keys = self.ros.map()
            for gh in self.ghosts:
                if gh["caught"] is not None:
                    continue
                gh["seen"] = not occluded(keys, view.pos, gh["pos"])
                gh["aimed"] = gh["seen"] and view.aim_angle_deg(gh["pos"]) < g["aim_deg"]
                gh["progress"] = gh["progress"] + dt if gh["aimed"] else max(0.0, gh["progress"] - 2 * dt)
                if gh["progress"] >= g["catch_seconds"]:
                    gh["caught"] = now
            if all(gh["caught"] is not None for gh in self.ghosts):
                self.hunt_time = now - self.phase_t
                self.enter("WIN")
        elif self.phase == "WIN" and now - self.phase_t > 12:
            self.start_hunt(view.pos)

    def draw_map(self, view):
        """The lidar map as the projector should see it, coloured by 0.5 m depth band. On the
        real room the colour steps sit on the real edges. Drawn at 1/8 resolution: each voxel
        is a square of its projected size, far ones first."""
        pts = unpack(self.ros.map())
        if len(pts) == 0:
            return
        k = 8
        w, h = self.size[0] // k, self.size[1] // k
        u, v, z = view.project(pts)
        ok = (z > 0.3) & (u > -40) & (u < self.size[0] + 40) & (v > -40) & (v < self.size[1] + 40)
        u, v, z = (u[ok] / k).astype(int), (v[ok] / k).astype(int), z[ok]
        half = np.clip(np.round(0.55 * VOXEL * view.f / z / k), 0, 6).astype(int)
        color = BANDS[(z / 0.5).astype(int) % len(BANDS)]
        img = np.zeros((w + 12, h + 12, 3), dtype=np.uint8)  # 6 px margin all round
        for r in range(7):  # small squares are the far voxels
            m = half == r
            if not m.any():
                continue
            order = np.argsort(-z[m])
            ur, vr, cr = u[m][order] + 6, v[m][order] + 6, color[m][order]
            inside = (ur >= r) & (ur < w + 12 - r) & (vr >= r) & (vr < h + 12 - r)
            ur, vr, cr = ur[inside], vr[inside], cr[inside]
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    img[ur + du, vr + dv] = cr
        small = pygame.surfarray.make_surface(img[6:-6, 6:-6])
        self.canvas.blit(pygame.transform.scale(small, self.size), (0, 0))

    def draw_ghost(self, view, gh, now):
        g = self.cfg["game"]
        age = 0.0 if gh["caught"] is None else now - gh["caught"]
        if age > 1.0 or not gh["seen"]:
            return
        bob = np.array([0, 0, 0.04 * math.sin(2.5 * now + gh["pos"][0] * 7)])
        u, v, z = view.project(gh["pos"] + bob)
        if not z[0] > 0.3:
            return
        size = int(np.clip(2 * g["ghost_radius"] * view.f / z[0] * (1 + age), 8, 2 * self.size[1]))
        img = pygame.transform.smoothscale(self.sprite, (size, int(size * 1.2)))
        if gh["caught"] is not None:
            img.fill((120, 255, 120, int(255 * (1 - age))), special_flags=pygame.BLEND_RGBA_MULT)
        c = (int(u[0]), int(v[0]))
        self.canvas.blit(img, img.get_rect(center=c))
        if gh["caught"] is None and gh["progress"] > 0:
            r = int(size * 0.75)
            end = 2 * math.pi * gh["progress"] / g["catch_seconds"]
            pygame.draw.arc(self.canvas, (80, 255, 80), (c[0] - r, c[1] - r, 2 * r, 2 * r), math.pi / 2 - end, math.pi / 2,
                            max(3, size // 12))

    def text(self, s, y, color=(255, 255, 255)):
        img = self.font.render(s, True, color)
        self.canvas.blit(img, img.get_rect(center=(self.size[0] // 2, int(y * self.size[1]))))

    def draw(self, view):
        now = time.monotonic()
        self.canvas.fill((0, 0, 0))
        w, h = self.size
        if view is None:
            self.text("Waiting for SLAM...", 0.5)
        else:
            if self.phase == "SCAN" or self.overlay or self.cfg["game"]["show_map"]:
                self.draw_map(view)
            for gh in self.ghosts:
                self.draw_ghost(view, gh, now)
            if self.phase == "SCAN":
                left = max(0, self.cfg["game"]["scan_seconds"] - (now - self.phase_t))
                self.text(f"Scan the room: look around  {left:.0f}", 0.12)
            elif self.phase == "HUNT":
                caught = sum(gh["caught"] is not None for gh in self.ghosts)
                self.text(f"{caught}/{len(self.ghosts)}   {now - self.phase_t:.0f} s", 0.08)
                r = int(view.f * math.tan(math.radians(self.cfg["game"]["aim_deg"])))
                pygame.draw.circle(self.canvas, (90, 90, 90), (w // 2, h // 2), r, 2)
            elif self.phase == "WIN":
                self.text(f"All ghosts caught in {self.hunt_time:.0f} s!", 0.5, (120, 255, 120))
        t = time.monotonic()
        pygame.display.flip()
        self.flip_s += time.monotonic() - t

    def report(self, view):
        colors = {"hidden": (1.0, 1.0, 1.0, 0.9), "aimed": (1.0, 0.9, 0.1, 1.0), "caught": (0.2, 1.0, 0.2, 0.5)}
        ghosts = []
        for gh in self.ghosts:
            state = "caught" if gh["caught"] is not None else "aimed" if gh["aimed"] else "hidden"
            ghosts.append((gh["pos"], colors[state], self.cfg["game"]["ghost_radius"]))
        caught = sum(gh["caught"] is not None for gh in self.ghosts)
        status = f"{self.phase} {caught}/{len(self.ghosts)} map={len(self.ros.map())} voxels"
        self.ros.set_view(ghosts, None if view is None else view.frustum(), status)
        return status

    def run(self):
        clock, last_slow, frames = pygame.time.Clock(), 0.0, 0
        keys = {pygame.K_r: "restart", pygame.K_s: "rescan", pygame.K_m: "map"}
        while True:
            dt = clock.tick(60) / 1000.0
            pose = self.ros.pose(self.cfg["projector"]["lead_s"])
            view = None if pose is None else View(self.cfg["projector"], self.size, pose)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key in (pygame.K_ESCAPE, pygame.K_q)):
                    return
                if ev.type == pygame.KEYDOWN and ev.key in keys:
                    self.ros.commands.append(keys[ev.key])
            while self.ros.commands:
                self.command(self.ros.commands.popleft(), None if view is None else view.pos)
            if view is not None:
                self.update(view, dt)
            self.draw(view)
            frames += 1
            now = time.monotonic()
            if now - last_slow > 1.0:  # once a second: config, markers, log
                self.load_config()
                print(f"{self.report(view)}  {frames / (now - last_slow):.0f} fps, flip {1000 * self.flip_s / frames:.0f} ms", flush=True)
                self.flip_s = 0.0
                if HEADLESS:
                    pygame.image.save(self.canvas, os.path.join(ROOT, "out", "frame.png"))
                last_slow, frames = now, 0
            elif frames % 6 == 0:
                self.report(view)


if __name__ == "__main__":
    # docker stop sends SIGTERM: leave through the QUIT event, and skip interpreter teardown,
    # where the still-spinning ROS thread aborts and leaves a core file.
    signal.signal(signal.SIGTERM, lambda *_: pygame.event.post(pygame.event.Event(pygame.QUIT)))
    Game().run()
    pygame.quit()
    os._exit(0)
