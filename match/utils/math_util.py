"""Collection of math-related helper functions."""

import torch


def fix_nan(tensor: torch.Tensor, default_value: float = 1.0) -> torch.Tensor:
  """Fix NaNs in tensor and replace them with the provided default value."""
  if torch.any(torch.isnan(tensor)):
    return torch.where(torch.isnan(tensor), default_value, tensor)
  else:
    return tensor


def euclidean_distance(
    pred_verts: torch.Tensor, gt_verts: torch.Tensor
) -> torch.Tensor:
  """Computes the euclidean distance between the predicted vertices and the target vertices.

  Args:
    pred_verts: predicted vertices, (..., 3)
    gt_verts: target vertices, (..., 3)

  Returns:
    euclidean_distance: euclidean distance between the predicted vertices and
    the
    target vertices, (...)
  """
  return torch.norm(pred_verts - gt_verts, dim=-1, p=2)


def _quaternion_to_matrix(quat: torch.Tensor) -> torch.Tensor:
  """Convert batched quaternion to rotation matrix.

  Args:
    quat: batched quaternions, dimension (B, 4).

  Returns:
    Rotation matrices, dimension (B, 3, 3).
  """
  # Unpack the quaterion.
  x = quat[:, 0]
  y = quat[:, 1]
  z = quat[:, 2]
  w = quat[:, 3]

  # Compute the main variables in the matrix. For details refer to:
  # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
  xx = x * x
  xy = x * y
  xz = x * z
  yy = y * y
  yz = y * z
  zz = z * z
  wx = w * x
  wy = w * y
  wz = w * z

  # Return the rotation matrices for these quaternions.
  rot_mat = torch.stack(
      [
          1.0 - 2.0 * (yy + zz),
          2.0 * (xy - wz),
          2.0 * (xz + wy),
          2.0 * (xy + wz),
          1.0 - 2.0 * (xx + zz),
          2.0 * (yz - wx),
          2.0 * (xz - wy),
          2.0 * (yz + wx),
          1.0 - 2.0 * (xx + yy),
      ],
      dim=-1,
  )
  return rot_mat.view([-1, 3, 3])


def sample_random_rotations(
    batch_size: int, num_rotations: int
) -> torch.Tensor:
  """Sample random rotations.

  Args:
    batch_size: batch size.
    num_rotations: number of rotations.

  Returns:
    Rotation matrices, dimension (batch_size, num_rotations, 3, 3).
  """
  quat = torch.randn(batch_size * num_rotations, 4, dtype=torch.float32)
  quat = torch.nn.functional.normalize(quat, dim=1)
  rotations = _quaternion_to_matrix(quat)
  return rotations.view([batch_size, num_rotations, 3, 3])
