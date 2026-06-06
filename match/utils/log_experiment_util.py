"""TensorBoard logging utility."""

import os
import einops
from match.utils import file_util
import tensorflow as tf
from torch import Tensor


def imgrid(tensor: Tensor, max_num: int = 4, max_view: int = 4):
  """Organize multi-view images in Dict `outputs` for wandb logging.

  Only process values in Dict `outputs` that have keys containing the word
  "images",
  which should be in the shape of (B, V, 3, H, W).
  """
  assert tensor.ndim == 5
  num, view = tensor.shape[:2]
  num, view = min(num, max_num), min(view, max_view)
  mvimages = einops.rearrange(tensor[:num, :view], "b v c h w -> c (b h) (v w)")
  return mvimages


class TensorBoardLogger:
  """TensorBoard logger."""

  _writer: tf.summary.SummaryWriter

  def __init__(self, log_path, accelerator=None):
    self.accelerator = accelerator
    if accelerator is not None and accelerator.is_main_process:
      file_util.makedirs(log_path, exist_ok=True)
      self._writer = tf.summary.create_file_writer(str(log_path))
      self._writer.init()
      self._writer.set_as_default()
    else:
      self._writer = None

  def add_scalar(self, name: str, value, step: int):
    if self.accelerator is not None and self.accelerator.is_main_process:
      tf.summary.scalar(name, value, step=step)

  def add_image(self, name: str, images, step: int):
    if self.accelerator is not None and self.accelerator.is_main_process:
      images = einops.rearrange(images.detach().cpu(), "c h w -> h w c")
      images = tf.convert_to_tensor(images)
      if tf.rank(images) == 3:
        images = tf.expand_dims(images, axis=0)
      tf.summary.image(name, images, step=step)

  def finish(self):
    if self._writer is not None:
      self._writer.close()

  def log(self, log_dict, step):
    if self.accelerator is not None and self.accelerator.is_main_process:
      for k, v in log_dict.items():
        if k.startswith("images"):
          v = imgrid(v)
          self.add_image(k, v, step)
        else:
          self.add_scalar(k, v, step)
