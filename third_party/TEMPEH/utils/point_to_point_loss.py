from torch import nn
import numpy.typing as npt
import torch


class WeightedP2PLoss(nn.Module):
  """P2P MSE loss with per-vertex weights """

  def __init__(
      self,
      num_points: int,
      vertex_groups: dict[str, npt.NDArray] | None = None,
      group_weights: dict[str, float] | None = None,
  ):
    super().__init__()

    groups = {} if vertex_groups is None else vertex_groups
    weights_cfg = {} if group_weights is None else dict(group_weights)

    default_weight = float(weights_cfg.pop('__default__', 1.0))
    if set(groups) != set(weights_cfg):
      raise ValueError(
          'vertex_masks and masks_weights must have same keys.'
      )

    vertex_weights = torch.full((num_points,), default_weight, dtype=torch.float32)
    for group_name, weight in weights_cfg.items():
      indices = torch.as_tensor(groups[group_name], dtype=torch.long)
      vertex_weights.index_fill_(0, indices, float(weight))

    self.register_buffer('_vertex_weights', vertex_weights, persistent=True)

  def forward(
      self,
      predicted_points: torch.Tensor,
      target_points: torch.Tensor,
      point_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:

    weights = self._vertex_weights.to(
        device=predicted_points.device,
        dtype=predicted_points.dtype,
    )

    if point_mask is not None:
      predicted_points = predicted_points[:, point_mask, :]
      target_points = target_points[:, point_mask, :]
      weights = weights[point_mask]

    squared_distances = (predicted_points - target_points).pow(2).sum(dim=-1)
    weighted_distances = squared_distances * weights.unsqueeze(0)
    return weighted_distances.mean(dim=1)

if __name__ == "__main__":
  import numpy as np
  num_points = 100
  vertex_groups = {
    "group1": np.arange(0, 10),
    "group2": np.arange(10, 20),
  }
  group_weights = {
    "group1": 1.0,
    "group2": 2.0,
  }
  loss = WeightedP2PLoss(num_points, vertex_groups, group_weights)
  loss = loss(torch.randn(100, 100, 3), torch.randn(100, 100, 3))
  print(loss)