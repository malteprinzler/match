import einops
import numpy as np
import numpy.typing as npt
import pudb
from match.utils import image_util, mesh_util, render_util
import tensorflow as tf
import torch
from typing import Tuple
import trimesh


def _map_range(
    values: tf.Tensor,
    from_min: tf.Tensor | float,
    from_max: tf.Tensor | float,
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



def bounding_box(mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        mask: (B, H, W) binary tensor
    Returns:
        boxes: (B, 4) long tensor [t, l, b, r]
               If a mask has no True, returns [-1, -1, -1, -1].
    """
    B, H, W = mask.shape

    # Make coordinate grids
    rows = torch.arange(H, device=mask.device).view(1, H, 1)
    cols = torch.arange(W, device=mask.device).view(1, 1, W)

    # Broadcast over mask
    masked_rows = torch.where(mask, rows, torch.full_like(rows, H))
    masked_cols = torch.where(mask, cols, torch.full_like(cols, W))

    t = masked_rows.amin(dim=(-2, -1))  # min row
    l = masked_cols.amin(dim=(-2, -1))  # min col

    masked_rows = torch.where(mask, rows, torch.full_like(rows, -1))
    masked_cols = torch.where(mask, cols, torch.full_like(cols, -1))

    b = masked_rows.amax(dim=(-2, -1))  # max row
    r = masked_cols.amax(dim=(-2, -1))  # max col

    boxes = torch.stack([t, l, b, r], dim=-1)  # (N, V, 4)

    # Handle no-True case
    has_true = mask.any(dim=(-2, -1))
    boxes[~has_true] = -1

    return boxes

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
  device = uv_renders.device
  dtype = uv_renders.dtype
  uv_renders = image_util.pad_image_to_fit_patchification(
      uv_renders, patch_size=img_patch_size, constant_value=-1.0
  )
  b, v, _, h, w = uv_renders.shape

  h_downsampled = int(h / downsample_for_computation)
  assert h_downsampled - h / downsample_for_computation == 0
  w_downsampled = int(w / downsample_for_computation)
  assert w_downsampled - w / downsample_for_computation == 0
  downsampled_patch_size = int(img_patch_size / downsample_for_computation)
  assert img_patch_size / downsample_for_computation == downsampled_patch_size

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
    uv_renders = uv_renders.to(device=device, dtype=dtype)
  n_imgpatches_v = h_downsampled // downsampled_patch_size
  n_imgpatches_h = w_downsampled // downsampled_patch_size

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

  # 2nd order scoring: virtually extending patches to contain uv region and calculate mask mean
  uv_patch_areas = einops.rearrange(uv_patch_masks.to(dtype), 'b v nv nh h w -> b v nv nh (h w)').sum(dim=-1)  # b v nv nh
  uv_patch_bboxs = einops.rearrange(
                      bounding_box(einops.rearrange(uv_patch_masks, 'b v nv nh h w -> (b v nv nh) h w')),
                      '(b v nv nh) c -> b v nv nh c', b=b, v=v, nv=sqrt_n_patches, nh=sqrt_n_patches)  # b v nv nh 4
  uv_patch_bboxs = einops.repeat(uv_patch_bboxs, 'b v nv_uv nh_uv c -> b v nv_uv nh_uv nv_img nh_img c', nv_img=n_imgpatches_v, nh_img=n_imgpatches_h)
  img_patch_bbox_l = torch.arange(0, w_downsampled, downsampled_patch_size, device=device) 
  img_patch_bbox_t = torch.arange(0, h_downsampled, downsampled_patch_size, device=device) 
  img_patch_bbox_l = einops.repeat(img_patch_bbox_l, 'nh_img -> b v nv_uv nh_uv nv_img nh_img', b=b, v=v, nv_uv=sqrt_n_patches, nh_uv=sqrt_n_patches, nv_img=n_imgpatches_v, nh_img=n_imgpatches_h)
  img_patch_bbox_t = einops.repeat(img_patch_bbox_t, 'nv_img -> b v nv_uv nh_uv nv_img nh_img', b=b, v=v, nv_uv=sqrt_n_patches, nh_uv=sqrt_n_patches, nv_img=n_imgpatches_v, nh_img=n_imgpatches_h)
  img_patch_bboxs = torch.stack((img_patch_bbox_t, img_patch_bbox_l, img_patch_bbox_t + downsampled_patch_size-1,  img_patch_bbox_l + downsampled_patch_size-1), dim=-1)
  union_bbxs = torch.stack(
      (
        torch.minimum(img_patch_bboxs[..., 0], uv_patch_bboxs[..., 0]),
        torch.minimum(img_patch_bboxs[..., 1], uv_patch_bboxs[..., 1]),
        torch.maximum(img_patch_bboxs[..., 2], uv_patch_bboxs[..., 2]),
        torch.maximum(img_patch_bboxs[..., 3], uv_patch_bboxs[..., 3]),
      ),
      dim=-1
    )
  union_areas = (union_bbxs[..., 2] - union_bbxs[..., 0] + 1) * (union_bbxs[..., 3] - union_bbxs[..., 1] + 1)
  secondary_scores = einops.rearrange(uv_patch_areas, 'b v nv_uv nh_uv -> b v nv_uv nh_uv 1 1') / union_areas
  secondary_scores = einops.rearrange(secondary_scores, 'b v nv_uv nh_uv nv_img nh_img -> b nv_uv nh_uv (v nv_img nh_img)')
  uv_img_patch_scores = torch.maximum(uv_img_patch_scores, secondary_scores*.1)  # downweighting secondary score to prioritize first order score


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

  uv_renders = render_util.render_mesh(
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


def get_uvgrid_barycentric_coords(
    template_triangle_uvs, uv_res: int = 256, eps: float = 1e-6
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
  """samples a grid of uv coordinates and returns the barycentric coordinates of the closest triangle for each uv coordinate.

  Args:
      face_uv_coords: uv texture coordinates of the gnome faces np.ndarray of
        shape (N_faces, 3, 2) normalized to [0, 1], origin at bottom left
      uv_res (int, optional): resolution of uv grid.
      eps (float, optional): small value for numerical stability when checking
        if point lies inside triangle.

  Returns:
      - uv grid coordinates as np.ndarray of shape (uv_res, uv_res, 2), ranging
      from 0...1, float, origin is on bottom left
      - barycentric coordinates of uv grid as np.ndarray of shape (uv_res,
      uv_res, 3), float
      - triangle indices of uv grid as np.ndarray of shape (uv_res, uv_res, 1),
      int
      - mask of uv grid as np.ndarray of shape (uv_res, uv_res, 1) indicating if
      grid point lies on any triangle, bool
      - barycentric coordinates of uv grid shifted by 1/uv_res in u direction.
      May be used for setting up a local coordinate system. np.ndarray of shape
      (uv_res, uv_res, 3)
      - barycentric coordinates of uv grid shifted by 1/uv_res in v direction.
      May be used for setting up a local coordinate system. np.ndarray of shape
      (uv_res, uv_res, 3)
  """

  # sampling uv coordinates of uv grid
  uv_coords = get_uvcoord_grid(uv_res, flip_y_axis=True)
  uv_coords = uv_coords.reshape(-1, 2)  # (H*W, 2)
  uv_coords_homg = np.hstack([uv_coords, np.ones_like(uv_coords[:, :1])])

  # generating uv mesh
  template_uv_verts = template_triangle_uvs.reshape(-1, 2)
  template_uv_verts_homg = np.hstack(
      [template_uv_verts, np.ones_like(template_uv_verts[:, :1])]
  )
  template_uv_faces = np.arange(template_uv_verts.shape[0]).reshape(-1, 3)
  template_uv_mesh = trimesh.Trimesh(
      vertices=template_uv_verts_homg,
      faces=template_uv_faces,
      process=False,
      validate=False,
  )

  # getting barycentric coordinates
  _, _, triangle_ids = trimesh.proximity.closest_point(
      template_uv_mesh, uv_coords_homg
  )
  bary = trimesh.triangles.points_to_barycentric(
      template_uv_mesh.triangles[triangle_ids], uv_coords_homg
  )
  mask = np.all(bary > -1 * eps, axis=1)

  # getting barycentric coordinates of local coordinate system
  # i.e. barycentric coordinates from points that are shifted by 1/uv_res in
  # each dimension
  uv_coordsys_u_homg = uv_coords_homg + np.array([[1 / uv_res, 0, 0]])
  uv_coordsys_v_homg = uv_coords_homg + np.array([[0, 1 / uv_res, 0]])
  bary_coordsys_u = trimesh.triangles.points_to_barycentric(
      template_uv_mesh.triangles[triangle_ids], uv_coordsys_u_homg
  )
  bary_coordsys_v = trimesh.triangles.points_to_barycentric(
      template_uv_mesh.triangles[triangle_ids], uv_coordsys_v_homg
  )

  # reshaping
  triangle_ids = triangle_ids.reshape(uv_res, uv_res, 1)
  bary = bary.reshape(uv_res, uv_res, 3)
  mask = mask.reshape(uv_res, uv_res, 1)
  uv_coords = uv_coords.reshape(uv_res, uv_res, 2)
  bary_coordsys_u = bary_coordsys_u.reshape(uv_res, uv_res, 3)
  bary_coordsys_v = bary_coordsys_v.reshape(uv_res, uv_res, 3)

  return uv_coords, bary, triangle_ids, mask, bary_coordsys_u, bary_coordsys_v


def get_uvcoord_grid(uv_res: int = 256, flip_y_axis: bool = False):
  """Returns a uv coordinate grid.

  Args:
    uv_res: resolution of the uv grid.
    flip_y_axis: if True, the y axis is flipped to origin at the bottom left.

  Return:
    uv_coords:A numpy array of shape (uv_res, uv_res, 2) containing the uv
      coordinates
      normalized from 0 ... 1. Coordinate of center of top-left pixel is (0,0)
      or (0,1) respectively.
  """
  uv_coords = (
      np.stack(np.meshgrid(np.arange(uv_res), np.arange(uv_res)), axis=-1)
  ) / (
      uv_res - 1
  )  # (H, W, 2)  uv coordinates 0 ... 1 with origin in the top left
  if flip_y_axis:
    uv_coords[..., 1] = (
        1 - uv_coords[..., 1]
    )  # flipping y axis to origin at bottom left
  return uv_coords
