"""
获取D435的内参矩阵和畸变系数,并保存为K.npy和dist.npy
"""

import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(
    rs.stream.color,
    640,
    480,
    rs.format.bgr8,
    30
)

profile = pipeline.start(config)

color_profile = profile.get_stream(
    rs.stream.color
).as_video_stream_profile()

intr = color_profile.get_intrinsics()

print("width =", intr.width)
print("height =", intr.height)

print("fx =", intr.fx)
print("fy =", intr.fy)

print("cx =", intr.ppx)
print("cy =", intr.ppy)

print("distortion =", intr.coeffs)

K = np.array([
    [intr.fx, 0, intr.ppx],
    [0, intr.fy, intr.ppy],
    [0, 0, 1]
], dtype=np.float64)

dist = np.array(
    intr.coeffs,
    dtype=np.float64
)

print("\nK =")
print(K)

print("\ndist =")
print(dist)

np.save("K.npy", K)
np.save("dist.npy", dist)

print("\nSaved:")
print("K.npy")
print("dist.npy")

pipeline.stop()