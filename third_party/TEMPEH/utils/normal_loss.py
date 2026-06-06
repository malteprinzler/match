import numpy as np
import numpy.typing as npt
from utils import mesh_helper
import torch
from torch import nn
import copy


class WeightedNormalLoss(nn.Module):
  """normal-alignment loss with per-vertex weighting."""

  def __init__(
      self,
      *,
      num_points: int,
      faces: torch.Tensor,
      vertex_groups: dict[str, npt.NDArray] | None = None,
      group_weights: dict[str, float] | None = None,
  ):
    super().__init__()
    groups = {} if vertex_groups is None else vertex_groups
    weights_by_group = {} if group_weights is None else copy.deepcopy(group_weights)
    baseline_weight = weights_by_group.pop('__default__', 1.0)

    if set(groups.keys()) != set(weights_by_group.keys()):
      raise ValueError(
          'vertex_groups and group_weights must have same keys.'
      )

    membership = np.zeros(num_points, dtype=np.bool_)
    for group_name, indices in groups.items():
      if np.any(membership[indices]):
        raise ValueError(
            'The vertex_groups are not disjoint. Group indices of group'
            f' {group_name} overlap with other masks.'
        )
      membership[indices] = True

    point_weights = np.full(num_points, baseline_weight, dtype=np.float32)
    for group_name, group_weight in weights_by_group.items():
      point_weights[groups[group_name]] = group_weight

    self._faces = faces
    self._vertex_weights = torch.from_numpy(point_weights)
    self._vertex_weights_sum = self._vertex_weights.sum()

    self.mesh_helper = mesh_helper.MeshHelper(num_vertices=num_points, faces=faces)

  def forward(
      self,
      predicted_points: torch.Tensor,
      target_points: torch.Tensor,
  ) -> torch.Tensor:
    device = predicted_points.device
    predicted_normals = self.mesh_helper.vertex_normals(
        vertices=predicted_points
    )
    target_normals = self.mesh_helper.vertex_normals(
        vertices=target_points
    )

    alignment = torch.sum(target_normals * predicted_normals, dim=-1)
    per_vertex_loss = 1.0 - alignment
    weights = self._vertex_weights.to(device)[None, :]
    normalized_weighted_loss = (per_vertex_loss * weights) / self._vertex_weights_sum
    return normalized_weighted_loss.sum(dim=-1)

