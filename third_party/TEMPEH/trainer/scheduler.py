from torch import optim
from torch.optim import lr_scheduler


class LinearConstantDecayScheduler(lr_scheduler.LambdaLR):

  def __init__(
      self,
      *,
      optimizer: optim.Optimizer,
      linear_steps: int,
      constant_steps: int,
      decay_steps: int,
      decay_rate: float,
      min_multiplier: float = 0.0,
  ):
    """Learning-rate multiplier with linear increase, plateau, then exponential decay.

    Schedule behavior:
    - steps ``[0, linear_steps)``: linear warmup from ``0`` to
      ``lr``.
    - next ``constant_steps`` steps: keep multiplier at ``lr``.
    - after that: decay by ``decay_rate`` in buckets of size ``decay_steps``
      (using rounded bucket index).
    - decay is lower-bounded by ``min_multiplier``.
    """
    self._linear_steps = linear_steps
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    self._min_multiplier = min_multiplier
    self._decay_begin_step = linear_steps + constant_steps

    super().__init__(optimizer=optimizer, lr_lambda=self._lr_multiplier)

  def _lr_multiplier(self, current_step: int) -> float:
    if current_step < self._linear_steps:
      return (current_step + 1) / float(self._linear_steps)

    if current_step > self._decay_begin_step:
      steps_since_decay_start = current_step - self._decay_begin_step
      decay_bucket = int(round(steps_since_decay_start / float(self._decay_steps)))
      return max(self._min_multiplier, self._decay_rate**decay_bucket)

    return 1.0
