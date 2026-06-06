from gtempeh_utils.geo_util import invert_c2w
import einops
import numpy as np
import numpy.typing as npt
import pudb
from gtempeh_utils import holobooth_camera
from gtempeh_utils import image_helper, mesh_util, render_helper
from datasets import data_utils
import tensorflow as tf
import torch
from typing import *


_IntArray = npt.NDArray[np.int32]
_FloatArray = npt.NDArray[np.float32]


def _map_range(
    values: tf.Tensor,
    from_min: Union[tf.Tensor, float],
    from_max: Union[tf.Tensor, float],
    to_min: float,
    to_max: float,
) -> tf.Tensor:
  """Remap values between [from_min, from_max] to [to_min, to_max]."""

  ones = tf.ones_like(values)
  from_range = from_max - from_min
  from_range = tf.where(from_range > 0.0, from_range, ones)
  values = (values - from_min) / from_range
  values = (values + to_min) * (to_max - to_min)
  return values


@torch.no_grad()
def get_img_uv_patch_correspondences(
    uv_renders: torch.Tensor,
    uv_res: int,
    uv_patch_size: int,
    img_patch_size: int,
    n_neighbors: int,
    downsample_for_computation: int = 1,
    return_uv_patch_mask: bool = True,
):
  """returns the correspondence mask between uv patches and image regions.

  Assuming center of bottom-left pixel of uvmap has coords (0,0) and center of
  top-right pixel has (1,1)

  Args:
    uv_renders: (B, V, 3, H, W)
    uv_res: resolution of uv map
    uv_patch_size: size of patches
    img_patch_size: size of patches
    n_neighbors: number of neighbors to consider for each uv patch
    downsample_for_computation: downsample uv map by this factor before
      computation for memory efficiency
    return_uv_patch_mask: if True, returns the uv patch mask (B, V, nv_uv,
      nh_uv, H, W), bool

  Returns:
    uv_patch_masks (optional): for every uv patch, contains visibility mask of
    all images
      (B, V, nv_uv, nh_uv, H, W), bool
    uv_img_patch_match_idcs: for every uv patch contains flattened idcs of top
      n_neighbors matching image patches.
      use unflatten_patch_idcs() to convert to respective unflattened idcs
      (B, nv_uv, nv_uv, n_neighbors), int
    uv_img_patch_match_scores: matching scores of top n_neighbor image patches
      (B, nv_uv, nv_uv, n_neighbors), float
  """
  uv_renders = image_helper.pad_image_to_fit_patchification(
      uv_renders, patch_size=img_patch_size, constant_value=-1.0
  )
  b, v, _, h, w = uv_renders.shape

  h_downsampled = int(h / downsample_for_computation)
  assert h_downsampled - h / downsample_for_computation == 0
  w_downsampled = int(w / downsample_for_computation)
  assert w_downsampled - w / downsample_for_computation == 0

  if downsample_for_computation > 1:
    uv_renders = einops.rearrange(
        torch.nn.functional.interpolate(
            einops.rearrange(uv_renders, "b v c h w -> (b v) c h w"),
            scale_factor=1.0 / downsample_for_computation,
            mode="nearest",
        ),
        "(b v) c h w -> b v c h w",
        b=b,
        v=v,
    )

  device = uv_renders.device
  dtype = uv_renders.dtype
  uv_pixel_size = 1 / (uv_res - 1)
  sqrt_n_patches = uv_res // uv_patch_size
  patch_edges_u = (
      -0.5 * uv_pixel_size
      + uv_pixel_size
      * uv_patch_size
      * torch.arange(sqrt_n_patches + 1, device=device, dtype=dtype)
  )  # (N_patches + 1)
  patch_edges_v = (
      1 - patch_edges_u
  )  # uv origin is at bottom left (sqrt_n_patches + 1)

  patch_edges_upper = patch_edges_v[:-1]  # (sqrt_n_patches,)
  patch_edges_lower = patch_edges_v[1:]
  patch_edges_left = patch_edges_u[:-1]
  patch_edges_right = patch_edges_u[1:]

  patch_edges_upper = einops.rearrange(patch_edges_upper, "n -> 1 n 1 1")
  patch_edges_lower = einops.rearrange(patch_edges_lower, "n -> 1 n 1 1")
  patch_edges_left = einops.rearrange(patch_edges_left, "n -> 1 n 1 1")
  patch_edges_right = einops.rearrange(patch_edges_right, "n -> 1 n 1 1")
  uv_renders = einops.rearrange(uv_renders, "b v c h w -> (b v) 1 c h w")
  uv_patch_mask_u = torch.logical_and(
      patch_edges_left <= uv_renders[:, :, 0],
      uv_renders[:, :, 0] < patch_edges_right,
  )  # (BV, sqrt_n_patches, h w)
  uv_patch_mask_v = torch.logical_and(
      patch_edges_lower <= uv_renders[:, :, 1],
      uv_renders[:, :, 1] < patch_edges_upper,
  )
  uv_patch_masks = torch.logical_and(
      einops.rearrange(uv_patch_mask_v, "bv n h w -> bv n 1 h w"),
      einops.rearrange(uv_patch_mask_u, "bv n h w -> bv 1 n h w"),
  )
  uv_patch_masks = torch.logical_and(
      uv_patch_masks,
      einops.rearrange(uv_renders[:, :, -1].bool(), "bv n h w -> bv n 1 h w"),
  )
  uv_patch_masks = einops.rearrange(
      uv_patch_masks, "(b v) nv nh h w -> b v nv nh h w", b=b, v=v
  )

  downsampled_patch_size = int(img_patch_size / downsample_for_computation)
  assert img_patch_size / downsample_for_computation == downsampled_patch_size
  uv_img_patch_scores = (
      einops.rearrange(
          uv_patch_masks,
          "b v nv_uv nh_uv (nv_img hpatch) (nh_img wpatch) -> b v nv_uv nh_uv"
          " nv_img nh_img (hpatch wpatch)",
          hpatch=downsampled_patch_size,
          wpatch=downsampled_patch_size,
      )
      .to(dtype)
      .mean(dim=-1)
  )  # for every uv patch, matching scores of image patches (B, V, nv, nh, nv_img, nh_img)

  uv_img_patch_scores = einops.rearrange(
      uv_img_patch_scores,
      "b v nv_uv nh_uv nv_img nh_img -> b nv_uv nh_uv (v nv_img nh_img)",
  )

  uv_img_patch_match_idcs = torch.argsort(
      uv_img_patch_scores, dim=-1, descending=True
  )[
      ..., :n_neighbors
  ]  # idcs (flattened) of top n_neighbors matching image patches (B, nv_uv, nh_uv, n_neighbors)

  uv_img_patch_match_scores = torch.gather(
      input=uv_img_patch_scores, index=uv_img_patch_match_idcs, dim=-1
  )  # (B, nv_uv, nh_uv, n_neighbors)

  if return_uv_patch_mask:
    uv_patch_masks = torch.repeat_interleave(
        uv_patch_masks, downsample_for_computation, dim=-1
    )
    uv_patch_masks = torch.repeat_interleave(
        uv_patch_masks, downsample_for_computation, dim=-2
    )
    return uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores
  else:
    return uv_img_patch_match_idcs, uv_img_patch_match_scores

  ###
  # old implementation, would in theory also allow for higher downsampling
  # factors than patch size
  ###

  # uv_img_patch_scores = (
  #     einops.rearrange(
  #         uv_patch_masks,
  #         "b v nv_uv nh_uv (nv_img hpatch) (nh_img wpatch) -> b v nv_uv nh_uv"
  #         " nv_img nh_img (hpatch wpatch)",
  #         hpatch=patch_size,
  #         wpatch=patch_size,
  #     )
  #     .to(dtype)
  #     .mean(dim=-1)
  # )  # for every uv patch, matching scores of image patches (B, V, nv, nh, nv_img, nh_img)

  # uv_img_patch_scores = einops.rearrange(
  #     uv_img_patch_scores,
  #     "b v nv_uv nh_uv nv_img nh_img -> b nv_uv nh_uv (v nv_img nh_img)",
  # )

  # uv_img_patch_match_idcs = torch.argsort(
  #     uv_img_patch_scores, dim=-1, descending=True
  # )[
  #     ..., : np.ceil(n_neighbors / n_outpatches_per_computedpatch)
  # ]  # idcs (flattened) of top n_neighbors matching image patches (B, nv_uv, nh_uv, n_neighbors/n_outpatches_per_computedpatch)

  # uv_img_patch_match_scores = torch.gather(
  #     input=uv_img_patch_scores, index=uv_img_patch_match_idcs, dim=-1
  # )  # (B, nv_uv, nh_uv, n_neighbors/n_outpatches_per_computedpatch)

  # # inflating neighbors to account for downsampling
  # if downsample_for_computation > 1:
  #   uv_img_patch_match_scores = torch.repeat_interleave(
  #       uv_img_patch_match_scores, n_outpatches_per_computedpatch, dim=-1
  #   )[..., :n_neighbors]
  #   uv_img_patch_match_idcs = upsample_patch_idcs(
  #       uv_img_patch_match_idcs,
  #       downsample_for_computation,
  #       h,
  #       w,
  #       patch_size,
  #   )
  # if return_uv_patch_mask:
  #   uv_patch_masks = torch.repeat_interleave(
  #       uv_patch_masks, downsample_for_computation, dim=-1
  #   )
  #   uv_patch_masks = torch.repeat_interleave(
  #       uv_patch_masks, downsample_for_computation, dim=-2
  #   )
  #   return uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores
  # else:
  #   return uv_img_patch_match_idcs, uv_img_patch_match_scores


def upsample_patch_idcs(
    idcs: torch.Tensor,
    upsample_factor: int,
    H: int,
    W: int,
    patch_size: int,
):
  """upsamples idcs to account for downsampling during get_img_uv_patch_correspondences()

  Args:
    idcs: correspondence idcs of shape idcs (flattened: V nv nh -> (V nv nh)) of
      top n_neighbors matching image patches. (N, n_neighbors)
    upsample_factor: upsample factor
    H: height of image
    W: width of image
    patch_size: size of patches

  Returns:
    upsampled_idcs: correspondence idcs upsampled to account for downsampling.
    (N, n_neighbors * downsample_for_computation**2)
  """
  upsample_arange = torch.arange(upsample_factor)
  n_outpatches_per_inpatch = upsample_factor**2
  h_downsampled = H // upsample_factor
  w_downsampled = W // upsample_factor
  vidx, row_idx, col_idx = unflatten_patch_idcs(
      idcs, h_downsampled, w_downsampled, patch_size
  )

  out_vidx = torch.repeat_interleave(
      vidx, n_outpatches_per_inpatch, dim=-1
  )  # (N, n_neighbors * downsample_for_computation**2)

  out_row_idx = einops.rearrange(
      row_idx, "n n_neighbors -> n n_neighbors 1 1"
  ) * upsample_factor + einops.rearrange(upsample_arange, "d -> 1 1 d 1")
  out_row_idx = torch.repeat_interleave(
      out_row_idx, upsample_factor, dim=-1
  )  # (N, n_neighbors, upsample_factor, upsample_factor)
  out_row_idx = einops.rearrange(
      out_row_idx, "n n_neighbors h w -> n (n_neighbors h w)"
  )

  out_col_idx = einops.rearrange(
      col_idx, "n n_neighbors -> n n_neighbors 1 1"
  ) * upsample_factor + einops.rearrange(upsample_arange, "d -> 1 1 1 d")
  out_col_idx = torch.repeat_interleave(
      out_col_idx, upsample_factor, dim=-2
  )  # (N, n_neighbors, upsample_factor, upsample_factor)
  out_col_idx = einops.rearrange(
      out_col_idx, "n n_neighbors h w -> n (n_neighbors h w)"
  )

  upsampled_idcs = flatten_patch_idcs(
      out_vidx, out_row_idx, out_col_idx, H=H, W=W, patch_size=patch_size
  )  # (N, n_neighbors * upsample_factor**2)

  return upsampled_idcs


def flatten_patch_idcs(
    vidx: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
):
  """inverse of unflatten_patch_idcs()

  Args:
    vidx: index of view
    row_idx: row index of patch in view
    col_idx: col index of patch in view
    H: height of image
    W: width of image
    patch_size: size of patches
  """

  img_patch_cols = W // patch_size
  img_patch_rows = H // patch_size
  idx = col_idx + img_patch_cols * (row_idx + img_patch_rows * vidx)
  return idx


def unflatten_patch_idcs(
    idcs: torch.Tensor,
    H: int,
    W: int,
    patch_size: int,
):
  """unflattens idcs of top n_neighbors matching image patches.

  Args:
    idcs: idcs (flattened) of tokens: V nv nh -> (V nv nh), tensor of arbitrary shape
    H: height of image
    W: width of image
    patch_size: size of patches

  Returns:
    vidx: index of view
    row_idx: row index of patch in view
    col_idx: col index of patch in view
  """
  img_patch_cols = W // patch_size
  img_patch_rows = H // patch_size
  vidx = idcs // (img_patch_rows * img_patch_cols)
  row_idx = (idcs - vidx * img_patch_cols * img_patch_rows) // img_patch_cols
  col_idx = idcs % img_patch_cols
  return vidx, row_idx, col_idx


def render_uvmaps(
    vertices: _FloatArray,
    faces: _IntArray,
    face_uv_coords: _FloatArray,
    camera_extrinsics: _FloatArray,
    camera_intrinsics: _FloatArray,
    camera_distortions: _FloatArray,
    image_size: Tuple[int, int],
    enable_cull_face: bool = False,
):

  return tf_render_uvmaps(
      vertices=tf.convert_to_tensor(vertices),
      faces=tf.convert_to_tensor(faces.astype(np.int32)),
      face_uv_coords=tf.convert_to_tensor(face_uv_coords),
      camera_extrinsics=tf.convert_to_tensor(camera_extrinsics),
      camera_intrinsics=tf.convert_to_tensor(camera_intrinsics),
      camera_distortions=tf.convert_to_tensor(camera_distortions),
      image_size=image_size,
      enable_cull_face=enable_cull_face,
  ).numpy()


def tf_render_uvmaps(
    vertices: tf.Tensor,
    faces: tf.Tensor,
    face_uv_coords: tf.Tensor,
    camera_extrinsics: tf.Tensor,
    camera_intrinsics: tf.Tensor,
    camera_distortions: tf.Tensor,
    image_size: Tuple[int, int],
    multisample_antialiasing: int = 1,
    enable_cull_face: bool = False,
) -> tf.Tensor:
  """Visualize a mesh by rasterizing it with TensorFlow Graphics.

  Args:
    vertices: vertices to be renderred, (B, V, 3).
    faces: mesh triangles, shared across the batch for batch rendering, (F, 3).
    face_uv_coords: uv coordinates of the mesh triangles, (F, 3, 2).
    camera_extrinsics: camera rotations and translations, (B, 3, 3).
    camera_intrinsics: camera intrinsics parameters, (B, 2, 3).
    camera_distortions: radial and tangential distortions, (B, 5).
    image_size: height and width of the output image.
    multisample_antialiasing: rendering with e.g., double resolution and then
      render with e.g. double resolution, and then downsample for anti-aliasing.
    background_color: color of the image background, (3,).
    enable_cull_face: flag if back facing triangles should be culled.
    vertex_colors: optional per-vertex colors, (B, V, 3).

  Returns:
    Rendered meshes, images of size (B, H, W, 3).
  """

  batch_size, num_points, _ = tf.unstack(tf.shape(vertices))

  projected_points = holobooth_camera.project_points(
      vertices,
      camera_extrinsics,
      camera_intrinsics,
      camera_distortions,
  )  # (1, N, 2)

  size = tf.cast(
      tf.expand_dims(tf.stack((image_size[1], image_size[0])), axis=0),
      dtype=tf.float32,
  )
  projected_points = 2.0 * projected_points / size - 1.0

  ones = tf.ones([batch_size, num_points, 1], dtype=vertices.dtype)
  points_homogeneous = tf.concat((vertices, ones), axis=-1)
  depths = camera_extrinsics[:, 2:3, :] @ tf.transpose(
      points_homogeneous, [0, 2, 1]
  )
  depths = tf.transpose(depths, [0, 2, 1])  # (1, N, 1)

  # Normalize depths to between 0.0 and 1.0.
  min_depth = tf.reduce_min(depths, axis=[1, 2], keepdims=True)
  max_depth = tf.reduce_max(depths, axis=[1, 2], keepdims=True)
  depths = _map_range(depths, min_depth, max_depth, 1e-3, 1.0 - 1e-3)

  # flipping triangles and uv coords to match rasterizer convention
  faces = tf.reverse(faces, [-1])
  face_uv_coords = tf.reverse(face_uv_coords, [-2])

  buffers = triangle_rasterizer.rasterize(
      vertices=tf.concat([projected_points, depths], axis=-1),
      triangles=faces,
      attributes={},
      view_projection_matrix=np.eye(4).astype(np.float32),
      image_size=(
          multisample_antialiasing * image_size[0],
          multisample_antialiasing * image_size[1],
      ),
      enable_cull_face=enable_cull_face,
      use_vectorized_map=True,
      backend=rasterization_backend.RasterizationBackends.CPU,
  )
  triangle_indices = buffers["triangle_indices"]  # (B, H, W, 1)
  barycentrics = buffers["barycentrics"]  # (B, H, W, 3)
  mask = buffers["mask"]  # (B, H, W, 1)
  pixel_face_uvcoords = tf.gather(
      face_uv_coords, triangle_indices[..., 0]
  )  # (B, H, W, 3, 2)
  pixel_uv_coords = tf.einsum(
      "bhwtc,bhwt->bhwc", pixel_face_uvcoords, barycentrics
  )  # (B, H, W, 2)
  pixel_uv_coords = tf.concat([pixel_uv_coords, mask], axis=-1)

  if multisample_antialiasing > 1:
    pixel_uv_coords = tf.image.resize(
        pixel_uv_coords, image_size, method=tf.image.ResizeMethod.AREA
    )

  return pixel_uv_coords


def render_uvmaps_from_TEMPEH_sample(sample, faces: np.ndarray,
    face_uv_coords: np.ndarray,
    use_gtverts: bool = False, out_height=None, out_width=None):

  # inverting procedure from data_utils.public_sample_to_TEMPEH_sample and factoring in public2gtempeh sample
  scale_factor = 1e-3

  imgs = sample['stereo_images']
  extrinsics = sample['stereo_camera_extrinsics'].clone()
  extrinsics = torch.cat([extrinsics, torch.zeros_like(extrinsics[..., :1, :])], dim=-2)
  extrinsics[..., -1, -1] = 1
  extrinsics[..., :3, :3] = extrinsics[..., :3, :3] / torch.det(extrinsics[..., :3, :3])[..., None, None]**(1/3)
  C2W = invert_c2w(extrinsics)

  verts = sample['v_pred'] * scale_factor

  intrinsics = sample['stereo_camera_intrinsics']
  B, V, C, H, W = imgs.shape
  fx = intrinsics[..., 0, 0]/W
  fy = intrinsics[..., 1, 1]/H
  cx = (intrinsics[..., 0, 2]+0.5)/W
  cy = (intrinsics[..., 1, 2]+0.5)/H
  fxfycxcy = torch.stack([fx,fy,cx,cy], dim=-1)

  gtempeh_sample = dict(
    image=imgs,
    C2W = C2W,
    stage1verts = verts,
    fxfycxcy=fxfycxcy
  )

  # resize gtempeh sample
  if out_height is not None:
    assert out_width is not None
    gtempeh_sample = data_utils.resize_gtempeh_sample(gtempeh_sample, [out_height, out_width])

  return render_uvmaps_from_sample(sample=gtempeh_sample, faces=faces, face_uv_coords=face_uv_coords, use_gtverts=use_gtverts)

def render_uvmaps_from_sample(
    sample,
    faces: np.ndarray,
    face_uv_coords: np.ndarray,
    use_gtverts: bool = False,
):
  """Renders uv maps for a given sample.

  Args:
    sample: data batch dict.
    faces: G-Nome triangles. np.ndarray of shape (F, 3)
    face_uv_coords: G-Nome triangles uv coordinates. np.ndarray of shape (F, 3,
      2)

  Returns:
    uv_renders: Rendered uv maps. np.ndarray of shape (B, V, H, W, 3), 0...1
  """
  c2w = sample["C2W"].cpu().numpy()  # (B, V, 4, 4)
  verts = (
      sample["verts" if use_gtverts else "stage1verts"].cpu().numpy()
  )  # (B, N, 3)
  b, v = c2w.shape[:2]
  h, w = sample["image"].shape[-2:]
  nverts = verts.shape[1]
  verts = np.broadcast_to(
      einops.rearrange(verts, "b n c -> b 1 n c"), (b, v, nverts, 3)
  )

  fxfycxcy = sample["fxfycxcy"].cpu().numpy()  # (B, V, 4)
  extrinsics = np.linalg.inv(c2w)[:, :, :3, :4]  # (B, V, 3, 4)
  intrinsics = np.zeros((b, v, 2, 3), dtype=np.float32)  # (B, V, 2, 3)
  intrinsics[:, :, 0, 0] = fxfycxcy[:, :, 0]
  intrinsics[:, :, 1, 1] = fxfycxcy[:, :, 1]
  intrinsics[:, :, 0, 2] = fxfycxcy[:, :, 2]
  intrinsics[:, :, 1, 2] = fxfycxcy[:, :, 3]
  intrinsics[:, :, 0] *= w
  intrinsics[:, :, 1] *= h

  distortions = np.zeros((b, v, 5), dtype=np.float32)

  # combine batch and view dimensions
  verts = einops.rearrange(verts, "b v n c -> (b v) n c")
  extrinsics = einops.rearrange(extrinsics, "b v c1 c2 -> (b v) c1 c2")
  intrinsics = einops.rearrange(intrinsics, "b v c1 c2 -> (b v) c1 c2")
  distortions = einops.rearrange(distortions, "b v c -> (b v) c")
  uv_mesh_verts, uv_mesh_faces, uv_mesh_vertuvcoords = mesh_util.get_uv_mesh(
      verts, faces, face_uv_coords
  )
  vertex_colors = einops.repeat(
      np.concatenate(
          (uv_mesh_vertuvcoords, np.ones_like(uv_mesh_vertuvcoords[..., :1])),
          axis=-1,
      ),
      "nverts c -> (b v) nverts c",
      b=b,
      v=v,
  )

  uv_renders = render_helper.render_holobooth_mesh(
      vertices=uv_mesh_verts,
      faces=uv_mesh_faces,
      camera_extrinsics=extrinsics,
      camera_intrinsics=intrinsics,
      camera_distortions=distortions,
      image_size=(h, w),
      enable_cull_face=False,
      vertex_colors=vertex_colors,
      flat_color=True,
      background_color=np.array([0, 0, 0], dtype=np.float32),
      antialias=False,
  )
  uv_renders = np.clip(
      einops.rearrange(uv_renders, "(b v) h w c -> b v h w c", b=b, v=v), 0, 1
  )
  return uv_renders
