# GOOGLE FILE
import time


class StepsTimer:
  """A timer that can be used to time steps in a loop."""

  def __init__(self, step: int = 0):
    self.reset(step)

  def reset(self, step: int):
    """Resets the timer."""
    self._time = time.time()
    self._step = step

  def steps_per_second(self, step: int) -> float:
    """Returns the number of steps per second sice the last timer reset."""
    elapsed_steps = step - self._step
    elapsed_time = time.time() - self._time
    return float(elapsed_steps) / elapsed_time
