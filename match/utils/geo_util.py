from typing import Optional, Tuple
import torch
from torch import Tensor
import torch.nn.functional as tF
import numpy as np
import einops
import dqtorch 
from torch.utils.data._utils.collate import default_collate
import numpy.typing as npt
from . import math_util


def collate_gaussians(gaussians):
  return default_collate(gaussians)


def apply_rotation_scale_center_to_points(points: np.ndarray, rotation:np.ndarray, scale: np.ndarray, center:np.ndarray):
        '''
        
        Args:
            points: ([B,] N,3)
            rotation: ([B,] 3, 3)
            scale: ([B,])
            center: ([B,] 3)
        
        Returns:
            rotated, scaled and centered points (N, 3)
        '''
        has_batch_dim = len(points.shape) == 3
        if not has_batch_dim:
           points = points[None]
           rotation = rotation[None]
           scale = scale[None]
           center = center[None]

        points = einops.einsum(rotation, points, 'b i j, b n j -> b n i')*einops.rearrange(scale, 'b -> b 1 1') - einops.rearrange(center, 'b c -> b 1 c')

        if not has_batch_dim:
           points = points[0]
        
        return points

def apply_inv_rotation_scale_center_to_points(points: np.ndarray, rotation:np.ndarray, scale: np.ndarray, center:np.ndarray):
        '''
        
        Args:
            points: ([B,] N,3)
            rotation: ([B,] 3, 3)
            scale: ([B,])
            center: ([B,] 3)
        
        Returns:
            rotated, scaled and centered points (N, 3)
        '''
        has_batch_dim = len(points.shape) == 3
        if not has_batch_dim:
           points = points[None]
           rotation = rotation[None]
           scale = scale[None]
           center = center[None]
        
        points = points + einops.rearrange(center, 'b c -> b 1 c')
        points = einops.einsum(rotation.swapaxes(-1,-2), points , 'b i j, b n j -> b n i')
        points = points / einops.rearrange(scale, 'b -> b 1 1')

        if not has_batch_dim:
           points = points[0]
        
        return points

def average_gaussians(gaussians):
  averaged_gaussians = dict()
  for k, v in gaussians.items():
    if k == 'rotation':
      B, V, C, H, W = v.shape
      v = einops.rearrange(v, 'b v c h w -> b (v h w) c')
      v = markley_average_quaternions(v)
      v = einops.rearrange(v, '(v h w) c -> v c h w', v=V, h=H, w=W)
    else:
      v = v.mean(dim=0)
    averaged_gaussians[k] = v
  return averaged_gaussians


def subtract_gaussians(ga, gb):
  '''
  for rotations 

  Gaussian feature maps are assumed to have shape (B, V, C, H, W)
  '''
  gd = dict()
  for k in ga:
    va = ga[k]
    vb = gb[k]
    if k == 'rotation':
      B, V, C, H, W = va.shape
      va = einops.rearrange(va, 'b v c h w -> b (v h w) c')
      vb = einops.rearrange(vb, 'b v c h w -> b (v h w) c')
      va = tF.normalize(va, dim=-1)
      vb = tF.normalize(vb, dim=-1)
      vb_conj = dqtorch.quaternion_conjugate(vb)
      if len(vb)<len(va):
        vb_conj = vb_conj.expand_as(va)
      if len(va)<len(vb):
        va = va.expand_as(vb_conj)
      vd = dqtorch.quaternion_mul(va.contiguous(), vb_conj.contiguous())
      vd = einops.rearrange(vd, 'b (v h w) c -> b v c h w', v=V, h=H, w=W)
    else:
      vd = va - vb
    gd[k] = vd
  return gd

def add_gaussians(ga, gb):
  '''
  for rotations 

  Gaussian feature maps are assumed to have shape (B, V, C, H, W)
  '''
  gd = dict()
  for k in ga:
    va = ga[k]
    vb = gb[k]
    if k == 'rotation':
      B, V, C, H, W = va.shape
      va = einops.rearrange(va, 'b v c h w -> b (v h w) c')
      vb = einops.rearrange(vb, 'b v c h w -> b (v h w) c')
      va = tF.normalize(va, dim=-1)
      vb = tF.normalize(vb, dim=-1)
      if len(vb)<len(va):
        vb = vb.expand_as(va)
      if len(va)<len(vb):
        va = va.expand_as(vb)
      vd = dqtorch.quaternion_mul(va.contiguous(), vb.contiguous())
      vd = einops.rearrange(vd, 'b (v h w) c -> b v c h w', v=V, h=H, w=W)
    elif k == 'opacity':
      vd = (va + vb).clip(0,1)
    else:
      vd = va + vb
    gd[k] = vd
  return gd



def markley_average_quaternions(quats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Markley average of quaternions across batches.

    Args:
        quats: Tensor of shape (B, N, 4), assumed normalized quaternions.
        eps: Small epsilon to avoid numerical issues.

    Returns:
        mean_quats: Tensor of shape (N, 4), normalized mean quaternions.
    """
    assert quats.ndim == 3 and quats.shape[-1] == 4, "Expected (B, N, 4)"
    B, N, _ = quats.shape

    # Normalize to be safe
    quats = quats / (quats.norm(dim=-1, keepdim=True).clamp_min(eps))

    # Align quaternion signs per N across batches to avoid hemisphere flips
    quats = align_quaternions(quats, quats[:1].expand_as(quats))

    # Compute Markley average per quaternion group N
    # For each N, we build the 4x4 covariance matrix across B
    Q = quats.transpose(0, 1)  # (N, B, 4)
    M = torch.einsum("nbi,nbj->nij", Q, Q) / B  # (N, 4, 4)

    # Compute principal eigenvector (largest eigenvalue) of M
    eigvals, eigvecs = torch.linalg.eigh(M)  # (N, 4), (N, 4, 4)
    mean_quats = eigvecs[..., -1]  # (N, 4): eigenvector with largest eigenvalue

    # Normalize to ensure unit quaternions
    mean_quats = mean_quats / (mean_quats.norm(dim=-1, keepdim=True).clamp_min(eps))
    mean_quats = mean_quats.to(quats)

    return mean_quats


def align_quaternions(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Flip quaternions so they are consistently oriented with a reference quaternions.
    Args:
        q: (...,4) tensor of quaternions
        ref: (...,4) reference quaternions
    """
    dot = (q * ref).sum(-1, keepdim=True)  # alignment score
    flip_mask = (dot < 0).float()
    return q * (1 - 2 * flip_mask)



def rigid_trafo_gaussians(gaussians, RT, ref_quat=None):
  '''
  
  Args:
    gaussians: dict with keys 'xyz' and 'rotation' with shapes (B, 1, C, H, W)
    RT: (B, 4, 4)
    ref_quat: rotation quaternions of RT will have same real sign as ref_quat (B, 4)
  '''

  trans_gaussians = dict()
  for k, v in gaussians.items(): 
    dtype = v.dtype
    if k == 'xyz':
      v = torch.cat((v, torch.ones_like(v[:, :, :1])), dim=2)
      v = einops.einsum(RT.to(v), v, 'b i j, b v j h w -> b v i h w')[:, :, :3]
    elif k == 'rotation':
      B, V, C, H, W = v.shape
      v = einops.rearrange(v, 'b v c h w -> b (v h w) c')
      rotquat = dqtorch.matrix_to_quaternion(RT[:, :3, :3].to(v))
      if ref_quat is not None:
        rotquat = align_quaternions(rotquat, ref_quat)
      rotquat = einops.rearrange(rotquat, 'b c -> b 1 c')
      rotquat = rotquat.expand_as(v)
      v = dqtorch.quaternion_mul(rotquat, v)
      v = einops.rearrange(v, 'b (v h w) c -> b v c h w', b=B, v=V, h=H, w=W)
    trans_gaussians[k] = v.to(dtype)
  return trans_gaussians


def random_rotation_matrix(angle_range: float):
    """
    Create a 3D rotation matrix for a random axis and a random angle.

    Args:
      angle_range: range of rotation angle in degree
    
    Returns:
        R (np.ndarray): (3, 3) rotation matrix
    """
    # Random axis (normalized)
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)

    # Random angle in degrees
    angle_deg = np.random.uniform(0, angle_range)
    angle_rad = np.deg2rad(angle_deg)

    # Rodrigues' rotation formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

    return R


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
  match axis:
    case 'x':
      rotation_matrices[:, 0, 0] = 1
      rotation_matrices[:, 1, 1] = np.cos(angles)
      rotation_matrices[:, 1, 2] = np.sin(angles)
      rotation_matrices[:, 2, 1] = -np.sin(angles)
      rotation_matrices[:, 2, 2] = np.cos(angles)

    case 'y':
      rotation_matrices[:, 0, 0] = np.cos(angles)
      rotation_matrices[:, 0, 2] = np.sin(angles)
      rotation_matrices[:, 1, 1] = 1
      rotation_matrices[:, 2, 0] = -np.sin(angles)
      rotation_matrices[:, 2, 2] = np.cos(angles)

    case 'z':
      rotation_matrices[:, 0, 0] = np.cos(angles)
      rotation_matrices[:, 0, 1] = np.sin(angles)
      rotation_matrices[:, 1, 0] = -np.sin(angles)
      rotation_matrices[:, 1, 1] = np.cos(angles)
      rotation_matrices[:, 2, 2] = 1
    
  return rotation_matrices

def invert_c2w(c2w: Tensor|np.ndarray) -> Tensor:
  """Inverts a camera to world transform.

  Args:
    c2w: The camera to world transform. (..., 4, 4)

  Returns:
    The inverted camera to world transform. (..., 4, 4)
  """
  w2c = torch.zeros_like(c2w) if isinstance(c2w, Tensor) else np.zeros_like(c2w)
  w2c[..., 3, 3] = 1
  w2c[..., :3, :3] = c2w[..., :3, :3].swapaxes(-1, -2)
  w2c[..., :3, -1:] = -c2w[..., :3, :3].swapaxes(-1, -2) @ c2w[..., :3, -1:]
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
   
###
# START GOOGLE CODE
###

_FloatArray = npt.NDArray[np.float32]
_EPSILON = 1e-7
_PROJECTION_CLIP_VALUE = 1e10


def get_camera_to_world_transformation(extrinsics: _FloatArray) -> _FloatArray:
  """Camera to world transformation.

  Transformation matrix that transforms points in the local camera
  coordinate systems to the world coordinate system.

  Args:
    extrinsics: batched camera entrinsics tensor, (B, 3, 4).

  Returns:
    Transformation matrix, (B, 3, 4).
  """
  scaled_rot = extrinsics[:, :3, :3]
  isotropic_scale = np.linalg.norm(scaled_rot[:, :, 0], axis=1, keepdims=True)
  isotropic_scale = isotropic_scale[:, :, np.newaxis]
  inv_rot = np.transpose(scaled_rot, [0, 2, 1]) / np.square(isotropic_scale)
  trans = extrinsics[:, :3, 3][:, :, np.newaxis]
  return np.concatenate((inv_rot, -inv_rot @ trans), axis=-1)



def get_camera_to_world_transformation(
    extrinsics: torch.Tensor,
) -> torch.Tensor:
  """Camera to world transformation.

  Transformation matrix that transforms points in the local camera
  coordinate systems to the world coordinate system.

  Args:
    extrinsics: batched camera entrinsics tensor, (B, 3, 4).

  Returns:
    Transformation matrix, (B, 3, 4).
  """
  scaled_rot = extrinsics[:, :3, :3]
  isotropic_scale = torch.linalg.norm(scaled_rot[:, :, 0], dim=1, keepdim=True)
  isotropic_scale = isotropic_scale.unsqueeze(dim=-1)
  inv_rot = scaled_rot.transpose(1, 2) / torch.square(isotropic_scale)
  trans = extrinsics[:, :3, 3].unsqueeze(dim=-1)
  return torch.cat((inv_rot, -inv_rot @ trans), dim=-1)


def project_points(
    *,
    points: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    distortions: torch.Tensor,
) -> torch.Tensor:
  """Perspective projection of points.

  Args:
    points: Points to be projected, (batch_size, num_points, 3).
    extrinsics: Batched camera rotations and translations, (batch_size, 3, 4).
    intrinsics: Batched camera intrinsics parameters, (batch_size, 2, 3).
    distortions: Batched radial and tangential distortions, (batch_size, 5).

  Returns:
    Batched projected points, (batch_size, num_points, 2).
  """
  batch_size, num_points, _ = points.shape
  device = points.device

  ones = torch.ones([batch_size, num_points, 1], dtype=points.dtype).to(device)
  points_homogeneous = torch.cat((points, ones), axis=-1)

  # Transformation from the world to the image coordinate system.
  points_image = extrinsics @ points_homogeneous.transpose(1, 2)
  points_image = points_image.transpose(1, 2)

  # Transformation to the undistorted image plane.
  z_coords = points_image[:, :, 2]
  z_coords = torch.where(z_coords.abs() < _EPSILON, 1.0, z_coords)

  points_image_x = points_image[:, :, 0] / z_coords
  points_image_y = points_image[:, :, 1] / z_coords

  k1, k2, k3 = distortions[:, 0], distortions[:, 1], distortions[:, 2]
  p1, p2 = distortions[:, 3], distortions[:, 4]
  r2 = points_image_x**2 + points_image_y**2
  r4 = r2**2
  r6 = r2 * r4
  radial_factor = 1.0 + k1[:, None] * r2 + k2[:, None] * r4 + k3[:, None] * r6

  xy = points_image[:, :, 0] * points_image[:, :, 1]
  # tangential_bias_x = 2*p1*xy + p2
  tangential_bias_x = 2.0 * p1[:, None] * xy + p2[:, None] * (
      r2 + 2.0 * points_image[:, :, 0] ** 2
  )
  # tangential_bias_y = p1*(r2 + 2*points_image[:,1]**2) + 2*p2*xy
  tangential_bias_y = 2.0 * p2[:, None] * xy + p1[:, None] * (
      r2 + 2.0 * points_image[:, :, 1] ** 2
  )

  radial_factor = math_util.fix_nan(radial_factor, default_value=1.0)
  tangential_bias_x = math_util.fix_nan(tangential_bias_x, default_value=0.0)
  tangential_bias_y = math_util.fix_nan(tangential_bias_y, default_value=0.0)

  points_image_x = points_image_x * radial_factor + tangential_bias_x
  points_image_y = points_image_y * radial_factor + tangential_bias_y
  points_image_z = torch.ones_like(points_image[:, :, 2]).to(device)
  points_image = torch.stack(
      (points_image_x, points_image_y, points_image_z), dim=-1
  )

  # Transformation from distorted image coordinates to the final image
  # coordinates with the camera intrinsics
  points_image = intrinsics @ points_image.transpose(1, 2)
  return points_image.transpose(1, 2)[:, :, :2]



###
# END GOOGLE CODE
###
