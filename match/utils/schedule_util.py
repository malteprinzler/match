"""Custom learning rate scheduler."""

# GOOGLE FILE

from torch import optim
from torch.optim import lr_scheduler


class WarmupDecayScheduler(lr_scheduler.LambdaLR):
  """Custom learning rate schedule."""

  def __init__(
      self,
      *,
      optimizer: optim.Optimizer,
      warmup_steps: int,
      constant_steps: int,
      decay_steps: int,
      decay_rate: float,
  ):
    """Applies exponential learning rate decay after a warmup period.

    The learning rate is ramped up linearly for the warmup steps, then held
    constant for the following fixed learning rate steps, and then exponentially
    decayed.

    Args:
      optimizer: The wrapped optimizer.
      warmup_steps: The number of steps to ramp up the learning rate.
      constant_steps: The number of steps to keep the learning rate constant
        after the warmup period.
      decay_steps: The number of steps to decay the learning rate.
      decay_rate: The rate of decay.
    """
    self._warmup_steps = warmup_steps
    self._constant_steps = constant_steps
    self._decay_steps = decay_steps
    self._decay_rate = decay_rate
    self._decay_start = warmup_steps + constant_steps

    # Initialize the base class with the learning rate lambda function.
    super().__init__(
        optimizer=optimizer,
        lr_lambda=self._get_lr_lambda,
    )

  def _get_lr_lambda(self, current_step: int):
    if current_step < self._warmup_steps:
      learning_rate_factor = (current_step + 1) / float(self._warmup_steps)
    elif current_step > self._decay_start:
      decay_step = current_step - self._decay_start
      decay_step = int(round(decay_step / float(self._decay_steps)))
      learning_rate_factor = self._decay_rate**decay_step
    else:
      learning_rate_factor = 1.0
    return learning_rate_factor
