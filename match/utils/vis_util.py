import io
from typing import Union
import einops
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import pudb
from match.utils import render_util, geo_util
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def pca_rgb(arr_hw_d: np.ndarray, eps: float = 1e-12):
    """
    arr_hw_d: array of shape (H, W, D) with arbitrary scale (not normalized)
    Returns: (H, W, 3) uint8 RGB image of the first 3 PCs
    """
    H, W, D = arr_hw_d.shape
    X = arr_hw_d.reshape(-1, D).astype(np.float64)      # (N, D), N = H*W

    # Center (mean subtraction) — required for PCA
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean

    # PCA via SVD of the centered data
    # Xc = U S Vt, rows=N samples, cols=D features
    # Principal directions are rows of Vt; scores = Xc @ Vt.T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T                                   # (N, D)

    # Take first 3 components (pad with zeros if D<3)
    K = min(3, D)
    pcs3 = np.zeros((X.shape[0], 3), dtype=np.float64)
    pcs3[:, :K] = scores[:, :K]

    # Scale each channel to 0..1 for display (min-max per channel)
    ch_min = pcs3.min(axis=0, keepdims=True)
    ch_max = pcs3.max(axis=0, keepdims=True)
    denom = np.maximum(ch_max - ch_min, eps)
    pcs3_norm = (pcs3 - ch_min) / denom                  # (N, 3) in [0,1]

    rgb = (pcs3_norm.reshape(H, W, 3) * 255).astype(np.uint8)
    return rgb

def apply_colormap(arr: np.ndarray, cmap_name="turbo", vmin=None, vmax=None):
    """
    Apply a matplotlib colormap to an array.

    Args:
        arr: (B, N) numpy array of scalar values
        cmap_name: string, name of colormap (e.g. "viridis", "plasma", "coolwarm")
        vmin: float, minimum value for normalization (default: arr.min())
        vmax: float, maximum value for normalization (default: arr.max())

    Returns:
        colors: (B, N, 3) numpy array with RGB values in [0, 1]
    """
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax

    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap = cm.get_cmap(cmap_name)

    # Normalize then map to colors
    return cmap(norm(arr))[..., :3]


def tensor_to_image(
    tensor: torch.Tensor, return_pil: bool = False
) -> Union[np.ndarray, Image.Image]:
  if tensor.ndim == 4:  # (B, C, H, W)
    tensor = einops.rearrange(tensor, "b c h w -> c h (b w)")
  assert tensor.ndim == 3  # (C, H, W)

  assert tensor.shape[0] in [1, 3]  # grayscale, RGB (not consider RGBA here)
  if tensor.shape[0] == 1:
    tensor = tensor.repeat(3, 1, 1)

  image = (tensor.permute(1, 2, 0).cpu().float().numpy() * 255).astype(
      np.uint8
  )  # (H, W, C)
  if return_pil:
    image = Image.fromarray(image)
  return image


def load_image(
    image_path: str, rgba: bool = False, imagenet_norm: bool = False
) -> torch.Tensor:
  with Image.open(image_path) as image:
    tensor_image = (
        torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    )  # (C, H, W) in [0, 1]

  if not rgba and tensor_image.shape[0] == 4:
    mask = tensor_image[3:4]
    tensor_image = tensor_image[:3] * mask + (1.0 - mask)  # white background

  if imagenet_norm:
    mean = torch.tensor(
        IMAGENET_MEAN, dtype=tensor_image.dtype, device=tensor_image.device
    ).view(3, 1, 1)
    std = torch.tensor(
        IMAGENET_STD, dtype=tensor_image.dtype, device=tensor_image.device
    ).view(3, 1, 1)
    tensor_image = (tensor_image - mean) / std

  return tensor_image  # (C, H, W)


ERROR_COLORSCALE_VMIN = 0.0 * 1000
ERROR_COLORSCALE_VMAX = 0.02 * 1000


def vis_3d_point_clouds(named_point_clouds: dict[str, np.ndarray], outpath: str):
    """
    Visualizes multiple 3D point clouds with different colors in a single 3D plot, and saves as HTML.

    Args:
        named_point_clouds: Dictionary mapping names to point clouds as np.ndarray of shape (N,3)
        outpath: Path to save the HTML visualization
    """
    fig = go.Figure()

    # Get a color palette large enough for all point clouds
    colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    
    for idx, (name, points) in enumerate(named_point_clouds.items()):
        color = colors[idx % num_colors]
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=name
        )
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Point Clouds',
        legend=dict(itemsizing='constant')
    )
    
    fig.write_html(outpath)
    print(f"Visualization saved to {outpath}")


def vis_3d_point_cloud_scores(
    points: np.ndarray,
    scores: np.ndarray,
    vmin=None,
    vmax=None,
    mesh_vertices: np.ndarray = None,
    mesh_faces: np.ndarray = None,
    mesh_opacity: float = 0.3,
    mesh_color: str = "lightgrey",
    show_text: bool = True,
):
    """
    Visualizes scores of 3D point clouds as a 3D heatmap scatter plot,
    with optional semi-transparent mesh overlay and per-point score text.

    Args:
        points: (N, 3) array of XYZ coordinates
        scores: (N,) array of per-point scalar scores
        vmin: minimum value for color scaling (default: min(scores))
        vmax: maximum value for color scaling (default: max(scores))
        mesh_vertices: (M, 3) array of mesh vertices
        mesh_faces: (F, 3) array of triangular faces (indices into vertices)
        mesh_opacity: float, transparency of mesh
        mesh_color: color of the mesh surface
        show_text: if True, display per-point scores as labels
    """
    assert points.shape[1] == 3, "points should have shape (N, 3)"
    assert points.shape[0] == scores.shape[0], "points and scores must have same length"

    vmin = np.min(scores) if vmin is None else vmin
    vmax = np.max(scores) if vmax is None else vmax

    text_labels = [f"{s:.2e}" for s in scores] if show_text else None

    # Scatter points with heatmap coloring
    scatter = go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers+text" if show_text else "markers",
        text=text_labels,
        textposition="top center",
        marker=dict(
            size=3,
            color=scores,
            colorscale="Viridis",
            cmin=vmin,
            cmax=vmax,
            colorbar=dict(title="Score"),
            opacity=0.8,
        )
    )

    data = [scatter]

    # Optional mesh overlay
    if mesh_vertices is not None and mesh_faces is not None:
        assert mesh_vertices.shape[1] == 3, "mesh_vertices must be (M, 3)"
        assert mesh_faces.shape[1] == 3, "mesh_faces must be (F, 3)"
        
        mesh = go.Mesh3d(
            x=mesh_vertices[:, 0],
            y=mesh_vertices[:, 1],
            z=mesh_vertices[:, 2],
            i=mesh_faces[:, 0],
            j=mesh_faces[:, 1],
            k=mesh_faces[:, 2],
            color=mesh_color,
            opacity=mesh_opacity,
            name="mesh",
        )
        data.append(mesh)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


def vis_3d_meshes(
    mesh_vertices: list[np.ndarray],
    mesh_faces: list[np.ndarray],
    mesh_names: list[str],
    mesh_opacity: float = 0.5,
):
    """
    Visualizes scores of 3D point clouds as a 3D heatmap scatter plot,
    with optional semi-transparent mesh overlay and per-point score text.

    Args:
        mesh_vertices: list of (M, 3) array of mesh vertices
        mesh_faces: list of (F, 3) array of triangular faces (indices into vertices)
        mesh_names: list of mesh names
        mesh_opacity: float, transparency of mesh
        show_text: if True, display per-point scores as labels
    """

    colors = px.colors.qualitative.Plotly
    num_colors = len(colors)

    data = list()
    for i in range(len(mesh_vertices)):
        
      mesh = go.Mesh3d(
          x=mesh_vertices[i][:, 0],
          y=mesh_vertices[i][:, 1],
          z=mesh_vertices[i][:, 2],
          i=mesh_faces[i][:, 0],
          j=mesh_faces[i][:, 1],
          k=mesh_faces[i][:, 2],
          color=colors[i%num_colors],
          opacity=mesh_opacity,
          name=mesh_names[i],
      )
      data.append(mesh)

    layout = go.Layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z"),
        ),
        margin=dict(l=0, r=0, b=0, t=0)
    )

    fig = go.Figure(data=data, layout=layout)
    return fig


def vis_vert_prediction(
    vert_pred: np.ndarray,
    vert_gt: np.ndarray,
    faces: np.ndarray,
    input_imgs: np.ndarray,
    figscale: float = 4.0,
    rot90= False,
    contrast = 1.5,
):
  """Visualizes the predicted verts and the ground truth verts.

  Args:
    vert_pred: Predicted vertices. (V, 3)
    vert_gt: Ground truth vertices. (V, 3)
    faces: G-Nome triangles. (F, 3)
    input_imgs: Input images. np.ndarray of shape (V, H, W, 3) [0...1].

  Returns:
    vis: Visualized vertices. np.ndarray of shape (B, H, W, 3) [0...255].
  """

  c2w = np.array([
          [1.0, 0.0, 0.0, 0.],
          [0.0, 1.0, 0.0, 0.],
          [0.0, 0.0, 1.0, -0.9],
          [0.0000, 0.0000, 0.0000, 1.0000],
      ],
      )
  h = 384
  w = 256
  fx, fy, cx, cy = 3.8017, 2.5336, 0.5000, 0.5000

  if rot90:
    c2w = np.array([
        [0.0, -1.0, 0.0, 0.],
        [1.0, 0.0, 0.0, 0.],
        [0.0, 0.0, 1.0, -0.9],
        [0.0000, 0.0000, 0.0000, 1.0000],
    ],
    )
    h, w = w, h
    fx, fy = fy, fx

  
  rotation_angles = np.array([-30., 0, 30])
  b = len(rotation_angles)
  rotation_matrices = geo_util.get_rotation_matrices(rotation_angles, 'y')

  c2ws = einops.einsum(rotation_matrices, c2w, 'b i j, j k -> b i k')  # (B, 4, 4)

  extrinsics = np.linalg.inv(c2ws)[:, :3, :4]
  intrinsics = np.array([[[fx, 0, cx], [0, fy, cy]]] * b)
  intrinsics[:, 0] *= w
  intrinsics[:, 1] *= h
  distortions = np.zeros((b, 5), dtype=np.float32)


  # center vertices
  center = np.mean(vert_gt, axis=0)
  vert_pred = vert_pred-center[None]
  vert_gt = vert_gt-center[None]

  vert_pred = np.tile(vert_pred[None], (b, 1, 1))
  vert_gt = np.tile(vert_gt[None], (b, 1, 1))

  pred_img = render_util.render_mesh(
      vertices=vert_pred.astype(np.float32),
      faces=faces.astype(np.int32),
      camera_extrinsics=extrinsics.astype(np.float32),
      camera_intrinsics=intrinsics.astype(np.float32),
      camera_distortions=distortions.astype(np.float32),
      image_size=(h, w),
      multisample_antialiasing=1,
      background_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
      enable_cull_face=False,
      contrast=contrast,
  )
  pred_img = np.clip(np.round(pred_img * 255.0), 0, 255).astype(np.uint8)
  if rot90:
    pred_img = np.rot90(pred_img, axes=(1, 2))

  gt_img = render_util.render_mesh(
      vertices=vert_gt.astype(np.float32),
      faces=faces.astype(np.int32),
      camera_extrinsics=extrinsics.astype(np.float32),
      camera_intrinsics=intrinsics.astype(np.float32),
      camera_distortions=distortions.astype(np.float32),
      image_size=(h, w),
      multisample_antialiasing=1,
      background_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
      enable_cull_face=False,
      contrast=contrast,
  )
  gt_img = np.clip(np.round(gt_img * 255.0), 0, 255).astype(np.uint8)
  if rot90:
    gt_img = np.rot90(gt_img, axes=(1, 2))

  vert_errors = np.linalg.norm((vert_pred - vert_gt) * 1000, ord=2, axis=-1)
  vert_mae = np.mean(vert_errors)
  vert_mse = np.mean(np.square((vert_pred - vert_gt) * 1000))
  cmap = plt.get_cmap("turbo")
  vert_error_colors = cmap(
      (vert_errors - ERROR_COLORSCALE_VMIN)
      / (ERROR_COLORSCALE_VMAX - ERROR_COLORSCALE_VMIN)
  )[..., :3]
  error_img = render_util.render_mesh(
      vertices=vert_gt.astype(np.float32),
      faces=faces.astype(np.int32),
      camera_extrinsics=extrinsics.astype(np.float32),
      camera_intrinsics=intrinsics.astype(np.float32),
      camera_distortions=distortions.astype(np.float32),
      image_size=(h, w),
      multisample_antialiasing=1,
      background_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
      enable_cull_face=False,
      vertex_colors=vert_error_colors.astype(np.float32),
  )
  error_img = np.clip(np.round(error_img * 255.0), 0, 255).astype(np.uint8)
  if rot90:
    error_img = np.rot90(error_img, axes=(1, 2))

  nrows = 4
  ncols = b
  fig, axes = plt.subplots(
      nrows, ncols, figsize=(ncols * figscale, nrows * figscale), squeeze=False
  )
  gs = axes[0, 0].get_gridspec()
  for ax in axes[0, :]:
    ax.remove()
  ax_input = fig.add_subplot(gs[0, :])
  if rot90:
    input_imgs = np.rot90(input_imgs, axes=(1, 2))
  ax_input.imshow(np.concatenate(input_imgs, axis=1))

  for i in range(b):
    axes[1, i].imshow(pred_img[i])
    axes[2, i].imshow(gt_img[i])
    axes[3, i].imshow(error_img[i])
  ax_input.set_ylabel("Input Image")
  axes[1, 0].set_ylabel("Predicted Verts")
  axes[2, 0].set_ylabel("Ground Truth Verts")
  axes[3, 0].set_ylabel("Vert Errors (MAE)")
  [(ax.set_xticks([]), ax.set_yticks([])) for ax in axes.flatten()]
  ax_input.set_xticks([])
  ax_input.set_yticks([])
  fig.suptitle(f"Vertex MAE: {vert_mae:.3e}, MSE: {vert_mse:.3e}")
  plt.tight_layout()
  img = fig2np(fig)
  plt.close(fig)
  return img


def vis_vertex_group_scores(
    vertex_scores: np.ndarray,
    vertex_group_weights: np.ndarray,
    vertex_group_names: list[str],
    figscale: float = 8.0,
):
  """Visualizes the vertex group scores.

  Args:
    vertex_group_scores: Vertex group scores. np.ndarray of shape (V)
    vertex_group_weights: Vertex group weights. np.ndarray of shape (G, V)
    vertex_group_names: Vertex group names. list of strings of length G
    figscale: Figure scale. float

  Returns:
    vis: Visualized vertex group scores. np.ndarray of shape (H, W, 3).
    [0...255]
  """

  group_scores = (
      vertex_group_weights @ vertex_scores
  ) / vertex_group_weights.sum(axis=1)
  sort_idcs = np.argsort(group_scores, axis=0)[::-1]
  group_scores = group_scores[sort_idcs]
  vertex_group_names = [vertex_group_names[i] for i in sort_idcs]

  fig, ax = plt.subplots(figsize=(figscale, figscale))
  ax.barh(np.arange(len(group_scores)), group_scores)
  for i, v in enumerate(group_scores):
    ax.text(
        v * 1.01, i, f"{v:.1f}", va="center", ha="left", fontsize=figscale * 0.9
    )
  ax.set_yticks(np.arange(len(group_scores)))
  ax.set_yticklabels(vertex_group_names)
  ax.set_xlabel("Group Scores")
  ax.invert_yaxis()
  ax.set_yticklabels(vertex_group_names, fontsize=figscale * 0.9)
  plt.tight_layout()
  img = fig2np(fig)
  plt.close(fig)
  return img


def vis_uv_scores(
    vertex_scores,
    faces,
    faces_uvcoords,
    figscale: float = 8.0,
    splat_size: float = 0.3,
    vmin: float | None = None,
    vmax: float | None = None,
    label: str = "Vertex Score",
):
  """Visualize vertex scores on uv grid.

  Args:
    vertex_scores: Vertex scores. np.ndarray of shape (V)
    faces: G-Nome triangles. np.ndarray of shape (F, 3)
    faces_uvcoords: G-Nome triangles uv coordinates. np.ndarray of shape (F, 3,
      2)
    uv_res: Resolution of the uv grid. int

  Returns:
    vis: Visualized vertex scores. np.ndarray of shape (H, W, 3). [0...255]
  """

  if vmin is None:
    vmin = np.min(vertex_scores)
  if vmax is None:
    vmax = np.percentile(vertex_scores, 95)

  pts = list()
  for f, f_uv in zip(faces, faces_uvcoords):
    for vid, uv_coord in zip(f, f_uv):
      pts.append([uv_coord[0], uv_coord[1], vertex_scores[vid]])
  pts = np.array(pts)

  fig = plt.figure(figsize=(figscale, figscale))
  plt.scatter(
      pts[:, 0],
      pts[:, 1],
      c=pts[:, 2],
      s=splat_size,
      cmap="turbo",
      vmin=vmin,
      vmax=vmax,
  )
  plt.colorbar(ticks=[vmin, vmax], label=label)
  img = fig2np(fig)
  plt.close(fig)
  return img


def plot_vertex_group(
    vertex_group_weights: np.ndarray,
    vertex_group_names: list[str],
    faces: np.ndarray,
    faces_uvcoords: np.ndarray,
    group: str,
    figscale: float = 8.0,
    splat_size: float = 0.3,
):
  """Plots the vertex group.

  Args:
    vertex_group_weights: Vertex group weights. np.ndarray of shape (G, V)
    vertex_group_names: Vertex group names. list of strings of length G
    faces: G-Nome triangles. np.ndarray of shape (F, 3)
    faces_uvcoords: G-Nome triangles uv coordinates. np.ndarray of shape (F, 3,
      2)
    group: Vertex group name. str

  Returns:
    img: Visualized vertex group. np.ndarray of shape (H, W, 3). [0...255]
  """

  group_idx = vertex_group_names.index(group)
  all_pts = list()
  group_pts = list()
  for f, f_uv in zip(faces, faces_uvcoords):
    for vid, uv_coord in zip(f, f_uv):
      all_pts.append([uv_coord[0], uv_coord[1]])
      if vertex_group_weights[group_idx, vid] > 0:
        group_pts.append([uv_coord[0], uv_coord[1]])
  all_pts = np.array(all_pts)
  group_pts = np.array(group_pts)
  fig, ax = plt.subplots(figsize=(figscale, figscale))
  ax.add_patch(
      plt.Rectangle(
          (0, 0),
          1,
          1,
          linewidth=1,
          edgecolor="r",
          facecolor="none",
          fill=False,
      )
  )
  plt.scatter(all_pts[:, 0], all_pts[:, 1], s=splat_size)
  plt.scatter(group_pts[:, 0], group_pts[:, 1], s=splat_size)
  plt.xlim(-0.1, 1.1)
  plt.ylim(-0.1, 1.1)
  img = fig2np(fig)
  plt.close(fig)
  return img


def turntable_vis(
    render_fn,
    C2W: torch.Tensor,
    rotation_center: tuple[float, float, float],
    phi_range_hor: tuple[float, float],
    phi_range_vert: tuple[float, float],
    nframes: int,
):
  """Renders turntable visualization.

  Args:
    render_fn: Function to render the turntable. Maps c2w (torch.Tensor, (4,4))
      to rendered image (torch.Tensor, (3, H, W))
    C2W: Camera to world transform. torch.Tensor of shape (4, 4)
    rotation_center: Rotation center. tuple of floats
    phi_range_hor: Horizontal rotation range. tuple of floats
    phi_range_vert: Vertical rotation range. tuple of floats
    nframes: Number of frames. int
  """

  subsection_frames = nframes // 4
  nframes = subsection_frames * 4
  rotation_center = np.array(rotation_center)
  phi_hor = []
  phi_vert = []

  # first turning
  phi_hor.append(
      np.linspace(phi_range_hor[0], phi_range_hor[1], subsection_frames)
  )
  phi_vert.append(
      np.zeros_like(phi_hor[-1]) + (phi_range_vert[0] + phi_range_vert[1]) * 0.5
  )

  # turning back
  phi_hor.append(
      np.linspace(phi_range_hor[1], phi_range_hor[0], subsection_frames)
  )
  phi_vert.append(
      np.zeros_like(phi_hor[-1]) + (phi_range_vert[0] + phi_range_vert[1]) * 0.5
  )

  # circular rotation
  psi = np.linspace(0, 2 * np.pi, 2 * subsection_frames)
  alpha_hor = np.cos(psi) * 0.5 + 0.5
  alpha_vert = np.sin(psi) * 0.5 + 0.5
  phi_hor.append(
      phi_range_hor[0] * alpha_hor + phi_range_hor[1] * (1 - alpha_hor)
  )
  phi_vert.append(
      phi_range_vert[0] * alpha_vert + phi_range_vert[1] * (1 - alpha_vert)
  )

  phi_hor = np.concatenate(phi_hor, axis=0)
  phi_vert = np.concatenate(phi_vert, axis=0)
  phi_hor_rd = phi_hor * np.pi / 180.0
  phi_vert_rd = phi_vert * np.pi / 180.0

  rot_hor = einops.repeat(np.eye(3), "H W -> B H W", B=nframes)
  rot_vert = einops.repeat(np.eye(3), "H W -> B H W", B=nframes)

  rot_hor[:, 0, 0] = np.cos(phi_hor_rd)
  rot_hor[:, 2, 2] = np.cos(phi_hor_rd)
  rot_hor[:, 0, 2] = -np.sin(phi_hor_rd)
  rot_hor[:, 2, 0] = np.sin(phi_hor_rd)

  rot_vert[:, 1, 1] = np.cos(phi_vert_rd)
  rot_vert[:, 2, 2] = np.cos(phi_vert_rd)
  rot_vert[:, 1, 2] = -np.sin(phi_vert_rd)
  rot_vert[:, 2, 1] = np.sin(phi_vert_rd)

  cam_rot = C2W[:3, :3].detach().cpu().numpy()
  cam_loc = C2W[:3, -1:].detach().cpu().numpy()

  cam_loc = (
      rot_hor @ rot_vert @ (cam_loc[None] - rotation_center[None, :, None])
      + rotation_center[None, :, None]
  )
  cam_rot = rot_hor @ rot_vert @ cam_rot

  C2W_turntable = einops.repeat(np.eye(4), "H W -> B H W", B=nframes)
  C2W_turntable[:, :3, :3] = cam_rot
  C2W_turntable[:, :3, -1:] = cam_loc
  C2W_turntable = torch.from_numpy(C2W_turntable).to(C2W)

  frames = []
  for i in range(nframes):
    frames.append(render_fn(C2W_turntable[i]))
  return torch.stack(frames, dim=0)


def fig2np(fig):
  """Convert a Matplotlib figure to a PIL Image and return it"""
  buf = io.BytesIO()
  fig.savefig(buf)
  buf.seek(0)
  with PIL.Image.open(buf) as img:
    img = np.array(img)[..., :3]
  return img