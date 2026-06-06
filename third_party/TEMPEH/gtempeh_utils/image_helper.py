"""Utilities for reading and writing images."""

import tempfile
import cv2
import imageio
import numpy as np
import numpy.typing as npt
import skimage
from gtempeh_utils import file_helper
import tensorflow as tf
import torch
from torch.nn import functional
from typing import *


_Array = npt.NDArray
_FloatArray = npt.NDArray[np.float32]


def increase_contrast(img:np.ndarray, factor:float=1.):
  '''
  Args:
    img: image with range 0...1
  '''
  midpoint = 0.5
  img_contrast = np.clip((img - midpoint) * factor + midpoint, 0, 1)
  return img_contrast

def load_image(image_path: file_helper.Path) -> _FloatArray:
  """Read image from file and return it as float array in [0.0, 1.0]."""
  with file_helper.open_file(image_path, 'rb') as f:
    image = imageio.imread(f)
  return np.array(image).astype(np.float32) / 255.0


def resize_image(
    image: _FloatArray, image_size: Tuple[int, int]
) -> _FloatArray:
  """Resize image to the specified image height and width."""
  return skimage.transform.resize(
      image, (image_size[0], image_size[1]), anti_aliasing=True
  )


def save_image(image_path: file_helper.Path, image: _FloatArray):
  """Save image to file."""
  image = np.clip(255.0 * image, 0.0, 255.0).astype(np.uint8)
  with file_helper.open_file(image_path, 'wb') as f:
    imageio.imsave(f, image, format=file_helper.get_extension(image_path))


def bilinearly_sample(
    feature_maps: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
  """Sample the feature maps though bilinear interpolation.

  Args:
    feature_maps: per-view feature maps, (B, H, W, F).
    points: projected 2D points, (B, N, 3).

  Returns:
    Sampled feature vector per points, (B, N, F).
  """
  _, height, width, _ = feature_maps.shape
  batch_size, num_points, _ = points.shape

  # Normalize the points to the range [-1, 1].
  u_coord = 2.0 * points[:, :, 0] / (width - 1.0) - 1.0
  v_coord = 2.0 * points[:, :, 1] / (height - 1.0) - 1.0
  grid2d_uv = torch.stack((u_coord, v_coord), dim=2).contiguous()
  grid2d_uv = grid2d_uv.view(batch_size, num_points, -1, 2)

  # Permute the feature maps to the order (B, F, H, W).
  feature_maps = torch.permute(feature_maps, [0, 3, 1, 2])
  feat2d_uv = functional.grid_sample(
      feature_maps,
      grid2d_uv,
      padding_mode='zeros',
      align_corners=True,
  )  # (B, F, N, 1).
  feat2d_uv = feat2d_uv.transpose(1, 3)
  return feat2d_uv.squeeze(1)


def nearest_sampling(
    feature_maps: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
  """Sample the feature maps though nearest sampling.

  Args:
    feature_maps: per-view feature maps, (B, H, W, F).
    points: projected 2D points, (B, N, 2).

  Returns:
    Sampled feature vector per points, (B, N, F).
  """
  _, height, width, _ = feature_maps.shape
  batch_size, num_points, _ = points.shape

  positions = points.contiguous().view([-1, 2])
  w_pos_int32 = torch.round(positions[:, 0]).to(torch.int32)
  w_pos_int32 = torch.clamp(w_pos_int32, min=0, max=width - 1)
  h_pos_int32 = torch.round(positions[:, 1]).to(torch.int32)
  h_pos_int32 = torch.clamp(h_pos_int32, min=0, max=height - 1)

  batch_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, num_points)
  batch_indices = batch_indices.contiguous().view([-1])
  sampled_features = feature_maps[batch_indices, h_pos_int32, w_pos_int32, :]
  return sampled_features.contiguous().view([batch_size, num_points, -1])


def encode_video(
    image_path: file_helper.Path,
    out_path: file_helper.Path,
    out_filename: str,
    image_ext: str,
    fps: int = 30,
):
  """Encode video from the image sequence."""
  img_files = file_helper.get_file_paths(image_path, file_ext=image_ext)
  if not img_files:
    raise ValueError(f'No image found in {image_path}')
  out_path.mkdir(parents=True, exist_ok=True)

  img_files.sort()
  images = [load_image(img_file) for img_file in img_files]
  fourcc = cv2.VideoWriter_fourcc(*'MP4V')
  h, w = images[0].shape[:2]
  with tempfile.NamedTemporaryFile(suffix='.mp4') as fv:
    video_out = cv2.VideoWriter(fv.name, fourcc, fps, (w, h))
    for img in images:
      img = (img * 255).astype(np.uint8)
      video_out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    video_out.release()

    video_out_path = out_path / f'{out_filename}.mp4'
    file_helper.copy(fv.name, video_out_path, overwrite=True)


def pad_image_to_fit_patchification(
    images: torch.Tensor, patch_size: int, constant_value: float = 0.0
):
  """Args:

    images: (..., H, W)
    patch_size: The patch size to pad for.
  Returns:
    images_padded: (..., H_padded, W_padded)
  """
  h_orig, w_orig = images.shape[-2:]

  # Calculate the padding needed to make image height and width to be a
  # multiple of the patch size.
  pad_h = (patch_size - h_orig % patch_size) % patch_size
  pad_w = (patch_size - w_orig % patch_size) % patch_size

  # Apply padding (padding_left, padding_right, padding_top, padding_bottom),
  # (B, C, H_padded, W_padded).
  return torch.nn.functional.pad(
      images, (0, pad_w, 0, pad_h), 'constant', constant_value
  )
