#!/usr/bin/env python3
"""Stand-in for the Airy + SLAM, to run the game with no sensor: a 6 x 5 x 2.5 m room and a
player at its centre, 1.2 m up, turning slowly. Publishes /slam/odom, /airy/imu, /slam/scan_preview.
"""
import math

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2, PointField

RATE = 0.4  # rad/s yaw


def room():
    g = np.arange(0.05, 6, 0.1)
    pts = []
    for z in np.arange(0.05, 2.5, 0.1):
        pts += [np.c_[g - 3, np.full_like(g, s * 2.5), np.full_like(g, z)] for s in (-1, 1)]
        h = g[g < 5]
        pts += [np.c_[np.full_like(h, s * 3.0), h - 2.5, np.full_like(h, z)] for s in (-1, 1)]
    x, y = np.meshgrid(g - 3, g[g < 5] - 2.5)
    pts.append(np.c_[x.ravel(), y.ravel(), np.zeros(x.size)])
    p = np.vstack(pts).astype(np.float32)
    return np.c_[p, np.ones(len(p), dtype=np.float32)]


class Fake(Node):
    def __init__(self):
        super().__init__("fake_slam")
        self.odom = self.create_publisher(Odometry, "/slam/odom", 5)
        self.imu = self.create_publisher(Imu, "/airy/imu", 50)
        self.scan = self.create_publisher(PointCloud2, "/slam/scan_preview", 1)
        self.cloud = room()
        self.t0 = self.get_clock().now()
        self.create_timer(0.005, self.on_imu)
        self.create_timer(0.1, self.on_odom)
        self.create_timer(0.5, self.on_scan)

    def on_imu(self):
        m = Imu()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "airy"
        m.angular_velocity.z = RATE
        self.imu.publish(m)

    def on_odom(self):
        now = self.get_clock().now()
        yaw = RATE * (now - self.t0).nanoseconds * 1e-9
        m = Odometry()
        m.header.stamp, m.header.frame_id, m.child_frame_id = now.to_msg(), "odom", "airy"
        m.pose.pose.position.z = 1.2
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.odom.publish(m)

    def on_scan(self):
        m = PointCloud2(height=1, width=len(self.cloud), point_step=16, row_step=16 * len(self.cloud), is_dense=True)
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), "odom"
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(("x", "y", "z", "intensity"))]
        m.data = self.cloud.tobytes()
        self.scan.publish(m)


rclpy.init()
rclpy.spin(Fake())
