"""Host side of the link to the ROS bridge container (bridge.py).

The game draws on the host, where SDL reaches the projector; ROS lives in a container. Between
them: pose packets and commands by UDP on localhost, the voxel map as a file in /dev/shm.
"""
import json
import os
import socket
import struct
import threading
import time
from collections import deque

import numpy as np

from .geom import rot_exp

POSE_PORT, VIEW_PORT = 47800, 47801  # bridge -> game, game -> bridge
MAP_FILE = "/dev/shm/ghost_hunt_map.npy"
POSE = struct.Struct("<16d")  # stamp, position 3, rotation 9 (row-major), gyro 3


class Link:
    def __init__(self):
        self.commands = deque()
        self._pose, self._pose_time = None, 0.0
        self._keys, self._map_mtime = np.zeros(0, dtype=np.int64), 0
        self._out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._in.bind(("127.0.0.1", POSE_PORT))
        threading.Thread(target=self._listen, daemon=True).start()

    def _listen(self):
        while True:
            data = self._in.recv(65536)
            if len(data) == POSE.size:
                self._pose, self._pose_time = POSE.unpack(data), time.monotonic()
            else:
                self.commands.append(data.decode(errors="replace").strip())

    def pose(self, lead_s=0.0):
        """(position, rotation) odom <- airy now, or None while no fresh SLAM pose arrives."""
        p = self._pose
        if p is None or time.monotonic() - self._pose_time > 1.0:
            return None
        return np.array(p[1:4]), np.array(p[4:13]).reshape(3, 3) @ rot_exp(np.array(p[13:16]) * lead_s)

    def map(self):
        """Sorted keys of every 10 cm voxel the scan has touched."""
        try:
            mtime = os.stat(MAP_FILE).st_mtime_ns
            if mtime != self._map_mtime:
                self._keys, self._map_mtime = np.load(MAP_FILE), mtime
        except (OSError, ValueError):
            pass
        return self._keys

    def _send(self, msg):
        self._out.sendto(json.dumps(msg).encode(), ("127.0.0.1", VIEW_PORT))

    def clear_map(self):
        self._keys = np.zeros(0, dtype=np.int64)
        self._send({"clear": True})

    def set_view(self, ghosts, frustum, status):
        """For Foxglove. ghosts: [(position, rgba, radius)], frustum: apex + 4 corners or None."""
        self._send({
            "ghosts": [[*map(float, p), *rgba, r] for p, rgba, r in ghosts],
            "frustum": None if frustum is None else np.round(frustum, 3).tolist(),
            "status": status,
        })
