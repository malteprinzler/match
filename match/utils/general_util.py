from argparse import Namespace
import os
from typing import Any, Dict, List, Optional

import accelerate as acc
import einops
import omegaconf
from match.utils import file_util
import torch
import numpy as np
import torchvision.transforms.functional as TF
import numpy.typing as npt

_IntArray = npt.NDArray[np.int32]
_FloatArray = npt.NDArray[np.float32]

def str_to_color(s: str) -> list[float]:
  """Converts a string to a list of 3 floats 0...1 representing a color."""
  if s == "white":
    return [1.0, 1.0, 1.0]
  elif s == "black":
    return [0.0, 0.0, 0.0]
  elif s == "random":
    return torch.rand(3).tolist()
  else:
    raise ValueError(f"Unknown color: {s}")

def torch_to_numpy(tensor: torch.Tensor) -> _IntArray | _FloatArray:
  """Converts a torch tensor to a numpy array."""
  return tensor.detach().cpu().numpy()

def cycle(iterable):
    """infinite looping over an iterable"""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

def method_parallelize_helper(args):
    obj, method_name, *method_args = args
    return getattr(obj ,method_name)(*method_args)


def resize_batch(images: np.ndarray, size: tuple[int, int], **kwargs) -> np.ndarray:
    """
    Resize a batch of images (N, 3, H, W) to (N, 3, H_new, W_new).
    
    Args:
        images: np.ndarray with shape (N, 3, H, W), values in [0, 255] or [0, 1].
        size: (H_new, W_new)
    """
    tensor = torch.from_numpy(images)  # (N, 3, H, W)
    resized = TF.resize(tensor, size, **kwargs)  # (N, 3, H_new, W_new)
    return resized.numpy()


def crop_tensor(x: torch.Tensor, crop_t: torch.Tensor, crop_l: torch.Tensor,
                crop_height: int, crop_width: int) -> torch.Tensor:
    """
    Crops a batched tensor.

    Assumes origin at top-left entry.

    Args:
        x: (B, C, H, W) tensor to crop
        crop_t: (B,) top coordinates of crops
        crop_l: (B,) left coordinates of crops
        crop_height: crop height
        crop_width: crop width

    Returns:
        Cropped tensor of shape (B, C, crop_height, crop_width)
    """
    B, C, H, W = x.shape
    device = x.device

    assert torch.all(crop_t >= 0) and torch.all(crop_t <= H - crop_height)
    assert torch.all(crop_l >= 0) and torch.all(crop_l <= W - crop_width)

    # Per-sample row and column offsets
    row_offsets = torch.arange(crop_height, device=device).view(1, crop_height, 1)  # (1, h, 1)
    col_offsets = torch.arange(crop_width, device=device).view(1, 1, crop_width)   # (1, 1, w)

    # Absolute coordinates
    rows = crop_t.view(B, 1, 1) + row_offsets  # (B, h, 1)
    cols = crop_l.view(B, 1, 1) + col_offsets  # (B, 1, w)

    # Broadcast to (B, h, w)
    rows = rows.expand(B, crop_height, crop_width)
    cols = cols.expand(B, crop_height, crop_width)

    # Batch indices
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(rows)

    # Advanced indexing
    cropped = x[batch_idx, :, rows, cols]  # (B, h, w, C)
    cropped = einops.rearrange(cropped, 'b h w c -> b c h w')

    return cropped


def crop_array(x: np.ndarray, crop_t: np.ndarray, crop_l: np.ndarray,
               crop_height: int, crop_width: int) -> np.ndarray:
    """
    Crops a batched array.

    Assumes origin at top-left entry.

    Args:
        x: (B, C, H, W) array to crop
        crop_t: (B,) top coordinates of crops
        crop_l: (B,) left coordinates of crops
        crop_height: crop height
        crop_width: crop width

    Returns:
        Cropped array of shape (B, C, crop_height, crop_width)
    """
    B, C, H, W = x.shape

    assert np.all(crop_t>=0) and np.all(crop_t<=H-crop_height)
    assert np.all(crop_l>=0) and np.all(crop_l<=W-crop_width)

    # Generate per-sample row and column indices
    row_offsets = np.arange(crop_height).reshape(1, crop_height, 1)  # (1, h, 1)
    col_offsets = np.arange(crop_width).reshape(1, 1, crop_width)    # (1, 1, w)

    # Broadcast crop_t and crop_l to build absolute coordinates
    rows = crop_t[:, None, None] + row_offsets  # (B, h, 1)
    cols = crop_l[:, None, None] + col_offsets  # (B, 1, w)

    # Broadcast to match crop shape
    rows = np.broadcast_to(rows, (B, crop_height, crop_width))  # (B, h, w)
    cols = np.broadcast_to(cols, (B, crop_height, crop_width))  # (B, h, w)

    # Build batch index array
    batch_idx = np.arange(B)[:, None, None]  # (B, 1, 1)

    # Advanced indexing: pick all channels for each batch
    cropped = x[batch_idx, :, rows, cols]  # (B, h, w, C)
    cropped = einops.rearrange(cropped, 'b h w c -> b c h w')

    return cropped


def split_into_chunks(lst, n):
    if n == 1:
      return [lst]
    else:
      k, m = divmod(len(lst), n)
      return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.types.Device
) -> dict[str, torch.Tensor]:
  out_batch = {}
  for key, value in batch.items():
    if isinstance(value, torch.Tensor):
      value = value.to(device)
    out_batch[key] = value
  return out_batch


def get_ckpt_path(ckpt_dir: str, ckpt_iter: int | None) -> str | None:
  """gets ckpt path

  ckpt_iter defines step of ckpt to load, if none, return None, if negative
  loads the latest checkpoint
  """
  if ckpt_iter is None:
    return None
  if ckpt_iter < 0:
    avail_ckpts = [x for x in file_util.list_dir(ckpt_dir) if x.isnumeric()]
    if not avail_ckpts:
      return None
    else:
      ckpt_iter = int(sorted(avail_ckpts)[ckpt_iter])

  return f"{ckpt_dir}/{ckpt_iter:06d}".replace("//", "/")


def load_ckpt(
    ckpt_path: str,
    model: Optional[torch.nn.Module] = None,
    accelerator: Optional[acc.Accelerator] = None,
    strict: bool = True,
) -> torch.nn.Module:
  if not os.path.exists(f"{ckpt_path}/zero_to_fp32.py"):
    acc.load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)
  else:  # from DeepSpeed
    if accelerator is not None:
      if accelerator.is_main_process:
        ensure_sysrun(
            f"python3 {ckpt_path}/zero_to_fp32.py"
            f" {ckpt_path} {ckpt_path} --safe_serialization"
        )
      accelerator.wait_for_everyone()  # wait before preparing checkpoints by the main process
    else:
      ensure_sysrun(
          f"python3 {ckpt_path}/zero_to_fp32.py"
          f" {ckpt_path} {ckpt_path} --safe_serialization"
      )
    acc.load_checkpoint_and_dispatch(model, ckpt_path, strict=strict)

  return model


def save_ckpt(ckpt_dir: str, ckpt_iter: int, hdfs_dir: Optional[str] = None):
  if hdfs_dir is not None:
    ensure_sysrun(
        f"tar -cf {ckpt_dir}/{ckpt_iter:06d}.tar -C {ckpt_dir} {ckpt_iter:06d}"
    )
    ensure_sysrun(f"hdfs dfs -put -f {ckpt_dir}/{ckpt_iter:06d}.tar {hdfs_dir}")
    ensure_sysrun(
        f"rm -rf {ckpt_dir}/{ckpt_iter:06d}.tar {ckpt_dir}/{ckpt_iter:06d}"
    )


def get_configs(
    yaml_path: str, cli_configs: List[str] = [], **kwargs
) -> omegaconf.DictConfig:
  yaml_configs = omegaconf.OmegaConf.load(yaml_path)
  cli_configs = omegaconf.OmegaConf.from_cli(cli_configs)

  configs = omegaconf.OmegaConf.merge(yaml_configs, cli_configs, kwargs)
  omegaconf.OmegaConf.resolve(configs)  # resolve ${...} placeholders
  return configs


def save_experiment_params(
    opt,
    save_dir: str,
) -> Dict[str, Any]:

  file_util.makedirs(save_dir, exist_ok=True)

  params = omegaconf.OmegaConf.create(vars(opt))

  with file_util.open_file(
      file_util.Path(save_dir, "params.yaml"), "w"
  ) as f:
    omegaconf.OmegaConf.save(params, f)
  return dict(params)


def save_model_architecture(model: torch.nn.Module, save_dir: str) -> None:
  file_util.makedirs(save_dir, exist_ok=True)

  num_buffers = sum(b.numel() for b in model.buffers())
  num_params = sum(p.numel() for p in model.parameters())
  num_trainable_params = sum(
      p.numel() for p in model.parameters() if p.requires_grad
  )
  message = (
      f"Number of buffers: {num_buffers}\n"
      + f"Number of trainable / all parameters: {num_trainable_params} /"
      f" {num_params}\n\n"
      + f"Model architecture:\n{model}"
  )

  with file_util.open_file(file_util.Path(save_dir, "model.txt"), "w") as f:
    f.write(message)


def ensure_sysrun(cmd: str):
  while True:
    result = os.system(cmd)
    if result == 0:
      break
    else:
      print(f"Retry running {cmd}")


def get_hdfs_files(hdfs_path: str) -> List[str]:
  lines = get_hdfs_lines(hdfs_path)
  if len(lines) == 0:
    raise ValueError(f"No files found in {hdfs_path}")

  return [line.split()[-1].split("/")[-1] for line in lines]


def get_hdfs_size(hdfs_path: str, unit: str = "B") -> int:
  lines = get_hdfs_lines(hdfs_path)
  if len(lines) == 0:
    raise ValueError(f"No files found in {hdfs_path}")

  byte_size = sum(int(line.split()[4]) for line in lines)
  if unit == "B":
    return int(byte_size)
  elif unit == "KB":
    return int(byte_size / 1024)
  elif unit == "MB":
    return int(byte_size / 1024 / 1024)
  elif unit == "GB":
    return int(byte_size / 1024 / 1024 / 1024)
  elif unit == "TB":
    return int(byte_size / 1024 / 1024 / 1024 / 1024)
  else:
    raise ValueError(f"Invalid unit: {unit}")


def get_hdfs_lines(hdfs_path: str) -> List[str]:
  return [
      line
      for line in os.popen(f"hdfs dfs -ls {hdfs_path}")
      .read()
      .strip()
      .split("\n")[1:]
  ]


def flag_values_to_namespace(flag_values) -> Namespace:
  """Converts absl.flags.FlagValues to argparse.Namespace."""
  namespace = Namespace()
  for name, flag in flag_values.flags_by_name.items():
    namespace.__setattr__(name, flag.value)
  return namespace


def weight_loss(
    loss: torch.Tensor,
    loss_weights: torch.Tensor,
    drop_zero_weights: bool = False,
):
  """weights loss

  Args:
    loss: loss tensor (B, ...)
    loss_weight: loss weight tensor (B, )
    drop_zero_weights: if True, drop the loss weights that are zero

  Returns:
    weighted loss as scalar tensor
  """
  loss_shape = loss.shape
  weight_target_shape = "B " + " ".join(
      ["1" for _ in range(len(loss_shape) - 1)]
  )
  loss_weights = einops.rearrange(loss_weights, "B -> " + weight_target_shape)

  if drop_zero_weights:
    non_zero_mask = loss_weights.flatten() != 0
    loss_weights = loss_weights[non_zero_mask]
    loss = loss[non_zero_mask]
  if len(loss) == 0:
    return torch.tensor(0.0, device=loss.device, dtype=loss.dtype)
  else:
    loss = loss * loss_weights
    loss = loss.mean()
    return loss
