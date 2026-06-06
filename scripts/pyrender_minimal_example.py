"""Minimal offscreen mesh rendering example with pyrender.

Run from the match repo root:
    python scripts/pyrender_minimal_example.py

On headless machines, PYOPENGL_PLATFORM must be set before pyrender/OpenGL
are imported. OSMesa (CPU software rasterizer) works without a display or GPU.

TensorFlow and pyrender cannot safely share a process on this setup:
importing TensorFlow before rendering segfaults inside r.render(). If you need
both, finish pyrender work first, delete the renderer, then import TensorFlow.
"""

import os
os.environ.pop('DISPLAY')
os.environ["PYOPENGL_PLATFORM"] = "egl"
# import tensorflow as tf


import numpy as np
import pyrender
import trimesh
from PIL import Image


def main():
    mesh = pyrender.Mesh.from_trimesh(trimesh.creation.icosphere(subdivisions=2))

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 1.0])
    scene.add(mesh)

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=512, viewport_height=512)
    try:
        color, depth = renderer.render(scene)
    finally:
        renderer.delete()

    out_path = "pyrender_minimal_example.png"
    Image.fromarray(color).save(out_path)
    print(f"Saved render to {out_path}")
    print(f"Color shape: {color.shape}, depth shape: {depth.shape}")


if __name__ == "__main__":
    main()
