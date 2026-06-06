import os
from typing import Tuple, Union
import einops
import torch


def rembg_and_center_wrapper(
    image_path: str,
    image_size: int,
    border_ratio: float,
    center: bool = True,
    model_name: str = "u2net",  # see https://github.com/danielgatis/rembg#models
) -> str:
  """Run `extensions/rembg_and_center.py` to remove background and center the image, and return the path to the new image."""
  os.system(
      f"python3 extensions/rembg_and_center.py {image_path}"
      + f" --size {image_size} --border_ratio {border_ratio} --model"
      f" {model_name}"
      + f" --center"
      if center
      else ""
  )
  directory, _ = os.path.split(image_path)
  file_base = os.path.basename(image_path).split(".")[0]
  new_filename = f"{file_base}_rgba.png"
  new_image_path = os.path.join(directory, new_filename)
  return new_image_path


def patchify(
    x: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]],
    tokenize: bool = True,
):
  if isinstance(patch_size, int):
    patch_size = (patch_size, patch_size)

  p1, p2 = patch_size
  if tokenize:
    return einops.rearrange(
        x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p1, p2=p2
    )
  else:
    return einops.rearrange(
        x, "b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=p1, p2=p2
    )


def unpatchify(
    x: torch.Tensor,
    patch_size: Union[int, Tuple[int, int]],
    input_size: Union[int, Tuple[int, int]],
    tokenize: bool = True,
):
  if isinstance(patch_size, int):
    patch_size = (patch_size, patch_size)
  if isinstance(input_size, int):
    input_size = (input_size, input_size)

  (p1, p2), (h, w) = patch_size, input_size
  if tokenize:
    return einops.rearrange(
        x, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)", h=h, w=w, p1=p1, p2=p2
    )
  else:
    return einops.rearrange(
        x, "b (p1 p2 c) h w -> b c (h p1) (w p2)", p1=p1, p2=p2
    )


def dilate(x: torch.Tensor, k: int):
  """Dilates the mask by k pixels.

  Args:

  - x: The mask to dilate. (B, 1, H, W)
  - k: number of pixels to dilate by.
  """
  if k == 0:
    return x
  weight = torch.ones(
      (1, 1, 2 * k + 1, 2 * k + 1), device=x.device, dtype=x.dtype
  )
  x = torch.nn.functional.conv_transpose2d(x, weight=weight, padding=k)
  return x
