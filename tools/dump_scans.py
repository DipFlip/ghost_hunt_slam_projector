#!/usr/bin/env python3
"""In the bridge container: N raw /airy/points scans (airy frame) and the SLAM pose before and
after, to /dev/shm/ghost_hunt_scans.npz. For calibration with the rig still."""
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2

n_scans = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rclpy.init()
node = Node("dump_scans")
scans, poses = [], []


def on_cloud(m):
    raw = np.frombuffer(m.data, dtype=np.uint8).reshape(-1, m.point_step)
    scans.append(raw[:, :16].copy().view(np.float32).reshape(-1, 4))


def on_odom(m):
    p, q = m.pose.pose.position, m.pose.pose.orientation
    poses.append([p.x, p.y, p.z, q.x, q.y, q.z, q.w])


node.create_subscription(PointCloud2, "/airy/points", on_cloud, QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT))
node.create_subscription(Odometry, "/slam/odom", on_odom, 5)
t = time.time()
while len(scans) < n_scans and time.time() - t < 20:
    rclpy.spin_once(node, timeout_sec=0.2)
np.savez_compressed("/dev/shm/ghost_hunt_scans.npz", points=np.vstack(scans), poses=np.array(poses))
print(len(scans), "scans", sum(map(len, scans)), "points; pose drift m", np.ptp(np.array(poses)[:, :3], axis=0).round(3))
