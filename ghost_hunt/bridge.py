"""ROS side, in a container: projector pose (SLAM odometry + gyro) and the voxel map out to
the game, the game's state back as markers for Foxglove. See link.py for the other end.

Everything is in FAST-LIO2's `odom` frame. rko_slam's loop-closure corrections are not applied:
inside one room the odometry drift is centimetres, and ghosts must not jump.
"""
import json
import os
import socket
import threading
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from .geom import VOXEL, pack, quat_to_rot, rot_exp
from .link import MAP_FILE, POSE, POSE_PORT, VIEW_PORT

MAP_RANGE = 10.0  # metres from the sensor; farther points are not added to the map
MIN_HITS = 4  # scans (2 Hz) a voxel must be seen in: people walking past do not enter the map


def stamp_s(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class Bridge(Node):
    def __init__(self):
        super().__init__("ghost_hunt")
        self.lock = threading.Lock()
        self.imu = deque(maxlen=600)  # (t, gyro), newest last
        self.odom_t = None
        self.pos = np.zeros(3)  # odom <- airy
        self.rot = np.eye(3)  # at the newest IMU sample
        self.map_keys = np.zeros(0, dtype=np.int64)  # sorted
        self.map_hits = np.zeros(0, dtype=np.int32)
        self.view = {"ghosts": [], "frustum": None, "status": ""}
        self.game = ("127.0.0.1", POSE_PORT)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        best_effort = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Imu, "/airy/imu", self.on_imu, best_effort)
        self.create_subscription(Odometry, "/slam/odom", self.on_odom, 5)
        self.create_subscription(
            PointCloud2, "/slam/scan_preview", self.on_scan, QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(
            String, "/ghost_hunt/cmd", lambda m: self.sock.sendto(m.data.encode(), self.game), 5)
        self.marker_pub = self.create_publisher(MarkerArray, "/ghost_hunt/markers", 2)
        self.status_pub = self.create_publisher(String, "/ghost_hunt/status", 2)
        self.create_timer(0.1, self.publish)
        self.save_map()
        threading.Thread(target=self.listen, daemon=True).start()

    # --- pose: the last SLAM pose, turned on by the gyro samples that came after it ---
    def on_imu(self, msg):
        w = msg.angular_velocity
        t, g = stamp_s(msg.header.stamp), np.array([w.x, w.y, w.z])
        with self.lock:
            if self.odom_t is None:
                self.imu.append((t, g))
                return
            if t > self.imu[-1][0] >= self.odom_t:
                self.rot = self.rot @ rot_exp(self.imu[-1][1] * (t - self.imu[-1][0]))
            self.imu.append((t, g))
            packet = POSE.pack(t, *self.pos, *self.rot.ravel(), *g)
        self.sock.sendto(packet, self.game)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        t0, rot = stamp_s(msg.header.stamp), quat_to_rot(q.x, q.y, q.z, q.w)
        with self.lock:
            prev = t0
            for t, g in self.imu:
                if t > t0:
                    rot = rot @ rot_exp(g * (t - prev))
                    prev = t
            self.odom_t, self.pos, self.rot = t0, np.array([p.x, p.y, p.z]), rot

    # --- map: every 10 cm voxel the scan has touched ---
    def on_scan(self, msg):
        n = msg.width * msg.height
        pts = np.frombuffer(msg.data, dtype=np.float32, count=n * 4).reshape(n, 4)[:, :3]
        with self.lock:
            pos = self.pos.copy()
        pts = pts[np.linalg.norm(pts - pos, axis=1) < MAP_RANGE]
        keys = np.unique(pack(np.floor(pts / VOXEL)))
        with self.lock:
            i = np.minimum(np.searchsorted(self.map_keys, keys), max(len(self.map_keys) - 1, 0))
            known = self.map_keys[i] == keys if len(self.map_keys) else np.zeros(len(keys), dtype=bool)
            self.map_hits[i[known]] += 1
            merged = np.concatenate([self.map_keys, keys[~known]])
            hits = np.concatenate([self.map_hits, np.ones((~known).sum(), dtype=np.int32)])
            order = np.argsort(merged)
            self.map_keys, self.map_hits = merged[order], hits[order]
        self.save_map()

    def save_map(self):
        with self.lock:
            np.save(MAP_FILE + ".tmp.npy", self.map_keys[self.map_hits >= MIN_HITS])
        os.replace(MAP_FILE + ".tmp.npy", MAP_FILE)

    # --- from the game ---
    def listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("127.0.0.1", VIEW_PORT))
        while True:
            msg = json.loads(sock.recv(65536))
            if msg.get("clear"):
                with self.lock:
                    self.map_keys, self.map_hits = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
                self.save_map()
            else:
                self.view = msg

    def publish(self):
        view, stamp = self.view, self.get_clock().now().to_msg()
        arr = MarkerArray()
        arr.markers.append(Marker(action=Marker.DELETEALL))
        for i, (x, y, z, r, g, b, a, radius) in enumerate(view["ghosts"]):
            m = Marker(ns="ghosts", id=i, type=Marker.SPHERE)
            m.header.frame_id, m.header.stamp = "odom", stamp
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2 * radius
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            arr.markers.append(m)
        if view["frustum"]:
            m = Marker(ns="projector", id=0, type=Marker.LINE_LIST)
            m.header.frame_id, m.header.stamp = "odom", stamp
            m.pose.orientation.w = 1.0
            m.scale.x = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = 1.0, 0.9, 0.2, 1.0
            f = view["frustum"]
            for a, b in ((0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)):
                for k in (a, b):
                    m.points.append(Point(x=f[k][0], y=f[k][1], z=f[k][2]))
            arr.markers.append(m)
        self.marker_pub.publish(arr)
        self.status_pub.publish(String(data=view["status"]))


if __name__ == "__main__":
    rclpy.init()
    rclpy.spin(Bridge())
