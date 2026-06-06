"""Rendering of a mesh with a given camera.."""

import einops
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pudb
import pyrender
import trimesh
from gtempeh_utils import image_helper
from typing import *

_IntArray = npt.NDArray[np.int32]
_FloatArray = npt.NDArray[np.float32]

_LIGHT_DIRECTIONS = (
    (1.0, 1.0, 1.0),
    (-1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0),
)
_DEFAULT_MESH_COLOR = (0.03, 0.5, 0.90)
_DEFAULT_LIGHT_INTENSITY = 2.0

_OFFSCREEN_RENDERERS = dict()


def render_holobooth_mesh(
    *,
    vertices: _FloatArray,
    faces: _IntArray,
    camera_extrinsics: _FloatArray,
    camera_intrinsics: _FloatArray,
    image_size: Tuple[int, int],
    camera_distortions: Optional[_FloatArray] = None,  # unused
    multisample_antialiasing: int = 1,  # unused
    background_color: Optional[_FloatArray] = None,
    enable_cull_face: bool = False,
    vertex_colors: Optional[_FloatArray] = None,
    face_colors: Optional[_FloatArray] = None,
    flat_color: bool = False,
    antialias: bool = True,
    contrast: float = 1.
) -> _FloatArray:
  """Visualizes a mesh by rasterizing it with pyrender.

  Camera Conventions: x: right, y: down, z: look-at

  Args:
    vertices: The vertices to be renderred, (B, V, 3).
    faces: The mesh triangles, shared across all batched vertices, (F, 3).
    camera_extrinsics: The camera rotations and translations, (B, 3, 4).
    camera_intrinsics: The camera intrinsics parameters, (B, 2, 3). In pixel
      units
    image_size: The height and width of the output image.
    multisample_antialiasing: Rendering size factor for anti-aliasing. A factor
      of 2 will render images at 4x the resolution (i.e., 2x the image size
      along each dimension) and then downsample for anti-aliasing.
    background_color: The color of the image background, (3,).
    vertex_colors: (optional) The per-vertex colors, (B, V, 3).

  Returns:
    The rendered meshes as images of size (B, H, W, 3).  0...1
  """
  batch_size = vertices.shape[0]
  height, width = image_size
  rendered_images = np.zeros((batch_size, height, width, 3), dtype=np.float32)

  background_color = (
      np.array([1.0, 1.0, 1.0], dtype=np.float32)
      if background_color is None
      else background_color
  )
  if len(background_color) == 3:
    background_color = np.concatenate([background_color, np.array([1.0])])

  if (face_colors is None) and (vertex_colors is None):
    vertex_colors = einops.repeat(
        np.array(_DEFAULT_MESH_COLOR),
        "c -> b v c",
        v=vertices.shape[1],
        b=batch_size,
    )

  for i in range(batch_size):
    scene = pyrender.Scene(bg_color=background_color)

    mesh = pyrender.Mesh.from_trimesh(
        trimesh.Trimesh(
            vertices=vertices[i],
            faces=faces,
            process=False,
            validate=False,
            vertex_colors=vertex_colors[i]
            if vertex_colors is not None
            else None,
            face_colors=face_colors[i] if face_colors is not None else None,
        )
    )
    scene.add(mesh)

    assert camera_intrinsics[i, 0, 1] == 0, "Skew not supported"
    camera_pose = np.linalg.inv(
        np.concatenate([camera_extrinsics[i], np.array([[0, 0, 0, 1]])], axis=0)
    )
    fx = camera_intrinsics[i, 0, 0]
    fy = camera_intrinsics[i, 1, 1]
    cx = camera_intrinsics[i, 0, 2]
    cy = camera_intrinsics[i, 1, 2]

    # # opencv to opengl conversion: z -> -z, y -> -y
    camera_pose = camera_pose @ np.diag(np.array([1, -1, -1, 1]))

    if False:  # visualize scene
      fig = plt.figure()
      ax = fig.add_subplot(111, projection="3d")

      # Plot camera
      camera_position = camera_pose[:3, 3]
      camera_axes = camera_pose[:3, :3]
      ax.quiver(
          camera_position[0],
          camera_position[1],
          camera_position[2],
          camera_axes[0, 0],
          camera_axes[1, 0],
          camera_axes[2, 0],
          color="r",
          label="Camera X",
      )
      ax.quiver(
          camera_position[0],
          camera_position[1],
          camera_position[2],
          camera_axes[0, 1],
          camera_axes[1, 1],
          camera_axes[2, 1],
          color="g",
          label="Camera Y",
      )
      ax.quiver(
          camera_position[0],
          camera_position[1],
          camera_position[2],
          camera_axes[0, 2],
          camera_axes[1, 2],
          camera_axes[2, 2],
          color="b",
          label="Camera Z",
      )

      # Plot mesh (subset of vertices)
      ax.scatter(
          vertices[i, ::10, 0],
          vertices[i, ::10, 1],
          vertices[i, ::10, 2],
          s=1,
          label="Mesh",
      )

      ax.set_xlabel("X")
      ax.set_ylabel("Y")
      ax.set_zlabel("Z")
      ax.legend()
      plt.show()

    camera = pyrender.camera.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        name="camera",
    )

    scene.add(camera, pose=camera_pose)
    for i_light, dir in enumerate(_LIGHT_DIRECTIONS):
      light = pyrender.DirectionalLight(
          name=f"light{i_light}",
          color=(1.0, 1.0, 1.0),
          intensity=_DEFAULT_LIGHT_INTENSITY,
      )
      pose = np.eye(4)
      pose[:3, 2] = -1 * np.array(dir)
      scene.add(light, pose=pose)

    # pyrender.Viewer(scene, use_raymond_lighting=True)
    viewport_height=image_size[0]
    viewport_width=image_size[1]
    antialias=antialias
    global _OFFSCREEN_RENDERERS
    renderer_key = f'{viewport_height}_{viewport_width}_{antialias}'
    if renderer_key in _OFFSCREEN_RENDERERS:
      r = _OFFSCREEN_RENDERERS[renderer_key]
    else:
      r = pyrender.OffscreenRenderer(
        viewport_height=image_size[0],
        viewport_width=image_size[1],
        antialias=antialias,
      )
      _OFFSCREEN_RENDERERS[renderer_key] = r
    flags = pyrender.constants.RenderFlags.NONE
    if flat_color:
      flags |= pyrender.constants.RenderFlags.FLAT
    if not enable_cull_face:
      flags |= pyrender.constants.RenderFlags.SKIP_CULL_FACES
    color, depth = r.render(scene, flags=flags)
    color =  color.astype(np.float32) / 255.0
    color = image_helper.increase_contrast(color, contrast)
    rendered_images[i] = color
    
  return rendered_images
