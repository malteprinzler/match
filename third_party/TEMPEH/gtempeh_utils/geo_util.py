from typing import Optional, Tuple, Union
import torch
from torch import Tensor
import torch.nn.functional as tF
import numpy as np
import einops


def get_rotation_matrices(angles: np.ndarray, axis:str) -> np.ndarray:
  '''
  
  Args:
    angles: rotation angles in degree (N,)
    axis: one of 'x', 'y', 'z'
  
  Returns:
    rotation matrix (4,4)
  '''

  assert axis in 'xyz'
  b = len(angles)
  angles = np.deg2rad(angles)

  rotation_matrices = np.zeros((b, 4, 4))
  rotation_matrices[:, 3, 3] = 1
  if axis == 'x':
      rotation_matrices[:, 0, 0] = 1
      rotation_matrices[:, 1, 1] = np.cos(angles)
      rotation_matrices[:, 1, 2] = np.sin(angles)
      rotation_matrices[:, 2, 1] = -np.sin(angles)
      rotation_matrices[:, 2, 2] = np.cos(angles)
  elif axis == 'y':
      rotation_matrices[:, 0, 0] = np.cos(angles)
      rotation_matrices[:, 0, 2] = np.sin(angles)
      rotation_matrices[:, 1, 1] = 1
      rotation_matrices[:, 2, 0] = -np.sin(angles)
      rotation_matrices[:, 2, 2] = np.cos(angles)
  elif axis == 'z':
      rotation_matrices[:, 0, 0] = np.cos(angles)
      rotation_matrices[:, 0, 1] = np.sin(angles)
      rotation_matrices[:, 1, 0] = -np.sin(angles)
      rotation_matrices[:, 1, 1] = np.cos(angles)
      rotation_matrices[:, 2, 2] = 1
  else:
    raise NotImplementedError()
  return rotation_matrices

def invert_c2w(c2w: Union[Tensor,np.ndarray]) -> Tensor:
  """Inverts a camera to world transform.

  Args:
    c2w: The camera to world transform. (..., 4, 4)

  Returns:
    The inverted camera to world transform. (..., 4, 4)
  """
  if isinstance(c2w, Tensor):
    if c2w.shape[-2] == 3:
      c2w = torch.cat((c2w, torch.zeros_like(c2w[..., 1:, :])), dim=-2)
      c2w[..., -1, -1] = 1
    w2c = torch.zeros_like(c2w)
    w2c[..., 3, 3] = 1
    w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
    w2c[..., :3, -1:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, -1:]
  elif isinstance(c2w, np.ndarray):
    if c2w.shape[-2] == 3:
      c2w = np.concatenate((c2w, np.zeros_like(c2w[..., 1:, :])), axis=-2)
      c2w[..., -1, -1] = 1
    w2c = np.zeros_like(c2w)
    w2c[..., 3, 3] = 1
    w2c[..., :3, :3] = c2w[..., :3, :3].swapaxes(-1, -2)
    w2c[..., :3, -1:] = -c2w[..., :3, :3].swapaxes(-1, -2) @ c2w[..., :3, -1:]
  else:
    raise NotImplementedError()
  return w2c



def normalize_normals(normals: Tensor, C2W: Tensor, i: int = 0) -> Tensor:
  """Normalize a batch of multi-view `normals` by the `i`-th view.

  Inputs:
      - `normals`: (B, V, 3, H, W)
      - `C2W`: (B, V, 4, 4)
      - `i`: the index of the view to normalize by

  Outputs:
      - `normalized_normals`: (B, V, 3, H, W)
  """
  _, _, R, C = C2W.shape  # (B, V, 4, 4)
  assert R == C == 4
  _, _, CC, _, _ = normals.shape  # (B, V, 3, H, W)
  assert CC == 3

  dtype = normals.dtype
  normals = normals.clone().float()
  transform = torch.inverse(C2W[:, i, :3, :3])  # (B, 3, 3)

  return torch.einsum("brc,bvchw->bvrhw", transform, normals).to(
      dtype
  )  # (B, V, 3, H, W)


def normalize_C2W(C2W: Tensor, i: int = 0, norm_radius: float = 0.0) -> Tensor:
  """Normalize a batch of multi-view `C2W` by the `i`-th view.

  Inputs:
      - `C2W`: (B, V, 4, 4)
      - `i`: the index of the view to normalize by
      - `norm_radius`: the normalization radius

  Outputs:
      - `normalized_C2W`: (B, V, 4, 4)
  """
  _, _, R, C = C2W.shape  # (B, V, 4, 4)
  assert R == C == 4

  device, dtype = C2W.device, C2W.dtype
  C2W = C2W.clone().float()

  if abs(norm_radius) > 0.0:
    radius = torch.norm(C2W[:, i, :3, 3], dim=1)  # (B,)
    C2W[:, :, :3, 3] *= norm_radius / radius.unsqueeze(1).unsqueeze(2)

  # The `i`-th view is normalized to a canonical matrix as the reference view
  transform = torch.tensor(
      [
          [1, 0, 0, 0],
          [0, 1, 0, 0],
          [0, 0, 1, norm_radius],
          [0, 0, 0, 1],  # canonical c2w in OpenGL world convention
      ],
      dtype=torch.float32,
      device=device,
  ) @ torch.inverse(
      C2W[:, i, ...]
  )  # (B, 4, 4)

  return (transform.unsqueeze(1) @ C2W).to(dtype)  # (B, V, 4, 4)


def unproject_depth(depth_map: Tensor, C2W: Tensor, fxfycxcy: Tensor) -> Tensor:
  """Unproject depth map to 3D world coordinate.

  Inputs:
      - `depth_map`: (B, V, H, W)
      - `C2W`: (B, V, 4, 4)
      - `fxfycxcy`: (B, V, 4)  in normalized screen coordinates [0, 1]

  Outputs:
      - `xyz_world`: (B, V, 3, H, W)
  """
  device, dtype = depth_map.device, depth_map.dtype
  B, V, H, W = depth_map.shape

  depth_map = depth_map.reshape(B * V, H, W).float()
  C2W = C2W.reshape(B * V, 4, 4).float()
  fxfycxcy = fxfycxcy.reshape(B * V, 4).float()
  K = torch.zeros(B * V, 3, 3, dtype=torch.float32, device=device)
  K[:, 0, 0] = fxfycxcy[:, 0]
  K[:, 1, 1] = fxfycxcy[:, 1]
  K[:, 0, 2] = fxfycxcy[:, 2]
  K[:, 1, 2] = fxfycxcy[:, 3]
  K[:, 2, 2] = 1

  y, x = torch.meshgrid(
      torch.arange(H), torch.arange(W), indexing="ij"
  )  # OpenCV/COLMAP camera convention
  y = y.to(device).unsqueeze(0).repeat(B * V, 1, 1) / (H - 1)
  x = x.to(device).unsqueeze(0).repeat(B * V, 1, 1) / (W - 1)
  # NOTE: To align with `plucker_ray(bug=False)`, should be:
  # y = (y.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / H
  # x = (x.to(device).unsqueeze(0).repeat(B*V, 1, 1) + 0.5) / W
  xyz_map = (
      torch.stack([x, y, torch.ones_like(x)], axis=-1) * depth_map[..., None]
  )
  xyz = xyz_map.view(B * V, -1, 3)

  # Get point positions in camera coordinate
  xyz = torch.matmul(xyz, torch.transpose(torch.inverse(K), 1, 2))
  xyz_map = xyz.view(B * V, H, W, 3)

  # Transform pts from camera to world coordinate
  xyz_homo = torch.ones((B * V, H, W, 4), device=device)
  xyz_homo[..., :3] = xyz_map
  xyz_world = torch.bmm(C2W, xyz_homo.reshape(B * V, -1, 4).permute(0, 2, 1))[
      :, :3, ...
  ].to(
      dtype
  )  # (B*V, 3, H*W)
  xyz_world = xyz_world.reshape(B, V, 3, H, W)
  return xyz_world

@torch.amp.autocast("cuda", torch.float32)
def plucker_ray(
    h: int, w: int, C2W: Tensor, fxfycxcy: Tensor
) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
  """Get Plucker ray embeddings.

  Coordinate conventions: 
    top left pixels center has coordinates (0.5, 0.5)
    bottom left pixels center has coordinates (W-.5, H-.5)

  Inputs:
      - `h`: image height
      - `w`: image width
      - `C2W`: (B, V, 4, 4)
      - `fxfycxcy`: (B, V, 4)

  Outputs:
      - `plucker`: (B, V, 6, `h`, `w`)
      - `ray_o`: (B, V, 3, `h`, `w`)
      - `ray_d`: (B, V, 3, `h`, `w`)
  """
  device, dtype = C2W.device, C2W.dtype
  B, V = C2W.shape[:2]

  C2W = C2W.reshape(B * V, 4, 4).float()
  fxfycxcy = fxfycxcy.reshape(B * V, 4).float()
  fx = fxfycxcy[:, 0]
  fy = fxfycxcy[:, 1]
  cx = fxfycxcy[:, 2]
  cy = fxfycxcy[:, 3]

  y_screen, x_screen = torch.meshgrid(
      torch.arange(h), torch.arange(w), indexing="ij"
  )  
  y_screen, x_screen = y_screen.to(device).flatten() + .5, x_screen.to(device).flatten() + .5  # OpenCV camera convention, top left pixel center is at 0.5,0.5, bottom right center at W-.5, H-.5
  x_cam = (x_screen[None] - cx[:, None] * w)/(fx[:, None]* w)  # (B*V, h*w)
  y_cam = (y_screen[None] - cy[:, None] * h)/(fy[:, None]* h)
  z_cam = torch.ones_like(x_cam)
  ray_d_cam = torch.stack([x_cam, y_cam, z_cam], dim=2)  # (B*V, h*w, 3)
  ray_d_world = einops.einsum(C2W[:, :3, :3], ray_d_cam, 'b i j, b n j -> b n i')  # (B*V, h*w, 3)
  ray_d_world = ray_d_world / torch.norm(ray_d_world, dim=2, keepdim=True)  # (B*V, h*w, 3)
  ray_o_world = C2W[:, :3, 3][:, None, :].expand_as(ray_d_world)  # (B*V, h*w, 3)

  ray_o_world = ray_o_world.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)  # (B V 3 H W)
  ray_d_world = ray_d_world.reshape(B, V, h, w, 3).permute(0, 1, 4, 2, 3)

  # # validation: projecting points back to screens and check if the coordinates match (should be .5 coordinates with stride 1)
  # test_points_world = einops.rearrange(ray_o_world + ray_d_world, 'b v c h w -> b (v h w) c')
  # test_c2w = einops.rearrange(C2W, '(b v) c1 c2 -> b v c1 c2', b=B, v=V)
  # test_fxfycxcy = einops.rearrange(fxfycxcy, '(b v) c1 -> b v c1', b=B, v=V)
  # test_projected_points = project_points_to_screen(points=test_points_world, c2w=test_c2w, fxfycxcy=test_fxfycxcy, H=h, W=w)
  # test_projected_points = einops.rearrange(test_projected_points, 'b v1 (v2 h w) c -> b v1 v2 h w c', v2=V, h=h, w=w)
  # for i_b in range(B):
  #   for i_v in range(V):
  #       print('max deviation x', torch.max(torch.abs(test_projected_points[i_b, i_v, i_v, ..., 0].flatten()- x_screen)))
  #       print('max deviation y', torch.max(torch.abs(test_projected_points[i_b, i_v, i_v, ..., 1].flatten()- y_screen)))


  plucker = torch.cat(
      [torch.cross(ray_o_world, ray_d_world, dim=2).to(dtype), ray_d_world.to(dtype)], dim=2
  )

  return plucker, (ray_o_world, ray_d_world)


def orbit_camera(
    elevs: Tensor,
    azims: Tensor,
    radius: Optional[Tensor] = None,
    is_degree: bool = True,
    target: Optional[Tensor] = None,
    opengl: bool = True,
) -> Tensor:
  """Construct a camera pose matrix orbiting a target with elevation & azimuth angle.

  Inputs:
      - `elevs`: (B,); elevation in (-90, 90), from +y to -y is (-90, 90)
      - `azims`: (B,); azimuth in (-180, 180), from +z to +x is (0, 90)
      - `radius`: (B,); camera radius; if None, default to 1.
      - `is_degree`: bool; whether the input angles are in degree
      - `target`: (B, 3); look-at target position
      - `opengl`: bool; whether to use OpenGL convention

  Outputs:
      - `C2W`: (B, 4, 4); camera pose matrix
  """
  device, dtype = elevs.device, elevs.dtype

  if radius is None:
    radius = torch.ones_like(elevs)
  assert elevs.shape == azims.shape == radius.shape
  if target is None:
    target = torch.zeros(elevs.shape[0], 3, device=device, dtype=dtype)

  if is_degree:
    elevs = torch.deg2rad(elevs)
    azims = torch.deg2rad(azims)

  x = radius * torch.cos(elevs) * torch.sin(azims)
  y = -radius * torch.sin(elevs)
  z = radius * torch.cos(elevs) * torch.cos(azims)

  camposes = torch.stack([x, y, z], dim=1) + target  # (B, 3)
  R = look_at(camposes, target, opengl=opengl)  # (B, 3, 3)
  C2W = torch.cat([R, camposes[:, :, None]], dim=2)  # (B, 3, 4)
  C2W = torch.cat([C2W, torch.zeros_like(C2W[:, :1, :])], dim=1)  # (B, 4, 4)
  C2W[:, 3, 3] = 1.0
  return C2W


def look_at(camposes: Tensor, targets: Tensor, opengl: bool = True) -> Tensor:
  """Construct batched pose rotation matrices by look-at.

  Inputs:
      - `camposes`: (B, 3); camera positions
      - `targets`: (B, 3); look-at targets
      - `opengl`: whether to use OpenGL convention

  Outputs:
      - `R`: (B, 3, 3); normalized camera pose rotation matrices
  """
  device, dtype = camposes.device, camposes.dtype

  if not opengl:  # OpenCV convention
    # forward is camera -> target
    forward_vectors = tF.normalize(targets - camposes, dim=-1)
    up_vectors = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)[
        None, :
    ].expand_as(forward_vectors)
    right_vectors = tF.normalize(
        torch.cross(forward_vectors, up_vectors), dim=-1
    )
    up_vectors = tF.normalize(
        torch.cross(right_vectors, forward_vectors), dim=-1
    )
  else:
    # forward is target -> camera
    forward_vectors = tF.normalize(camposes - targets, dim=-1)
    up_vectors = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)[
        None, :
    ].expand_as(forward_vectors)
    right_vectors = tF.normalize(
        torch.cross(up_vectors, forward_vectors), dim=-1
    )
    up_vectors = tF.normalize(
        torch.cross(forward_vectors, right_vectors), dim=-1
    )

  R = torch.stack([right_vectors, up_vectors, forward_vectors], dim=-1)
  return R


@torch.amp.autocast("cuda", torch.float32)
def project_points_to_screen(points: torch.Tensor, c2w: torch.Tensor, fxfycxcy: torch.Tensor, H:int, W:int):
  """
  projects points from world to screen
  
  Args:
    points: (B, N, 3)
    c2w: (B, V, 4, 4)
    fxfycxcy: (B, V, 4)
  Returns:
    projected points ranging from 0 (top left corner of top left pixel) to H, W (bottom right corner f bottom right pixel) + depth channel, (B, V, N, 3)
  """
  # Backproject vertices
  fx, fy, cx, cy = torch.unbind(fxfycxcy, dim=-1)
  fx = fx * W # (B, V)
  fy = fy * H
  cx = cx * W
  cy = cy * H

  w2c = invert_c2w(c2w)


  points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
  points_cam = einops.einsum(w2c, points, "b v i j, b n j -> b v n i")
  point_x_2d = fx[..., None] * points_cam[..., 0] / points_cam[..., 2] + cx[..., None]
  point_y_2d = fy[..., None] * points_cam[..., 1] / points_cam[..., 2] + cy[..., None]
  points_screen = (torch.stack([point_x_2d, point_y_2d, points_cam[..., 2]], dim=-1))  # (B, V, N, 3)


  return points_screen
   
