import os
import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pudb
from match.utils import file_util, general_util
import tensorflow as tf
import torch
from match.utils import geo_util, mesh_util
import math
from zipp import Path as ZipPath
from typing import List
from torch.utils.data import IterableDataset
import tqdm
from match.utils import vis_util
import random
import torchvision
from typing import Dict
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random
import copy
import torch.nn.functional as tF

AVA_TEMPLATE_MESH_INFO = mesh_util.load_obj("assets/ava256/face_topology_cleaned.obj")


def scale_crop(img, crop_size, h_offset, w_offset, scale_factor, K=None):
    """Apply per-image scaling and cropping (with optional padding) on batched
    torch tensors.

    Args:
        img: torch.Tensor with shape (B, C, H, W)
        crop_size: tuple (crop_h, crop_w) or int (assuming square crop)
        h_offset, w_offset: per-image crop offsets, shape (B,)
        scale_factor: per-image scale factors, shape (B,), >1.0 enlarges, <1.0 shrinks
        K: optional camera intrinsics, torch.Tensor or np.ndarray with
           shape (3, 3) or (B, 3, 3)
        debug: unused (kept for API compatibility)
        debug_root: unused (kept for API compatibility)

    Returns:
        img_aug: torch.Tensor with shape (B, C, crop_h, crop_w)
        K_aug (if K provided): intrinsics with same batch shape as input K
    """
    if not torch.is_tensor(img):
        raise RuntimeError(f"scale_crop expects torch.Tensor input, got {type(img)}")

    if img.dim() != 4:
        raise RuntimeError(f"scale_crop expects img with shape (B, C, H, W), got shape {img.shape}")

    B, C, H, W = img.shape

    if isinstance(crop_size, tuple):
        crop_height, crop_width = crop_size
    else:
        crop_width = crop_height = int(crop_size)

    device = img.device
    dtype = img.dtype

    def _to_1d_tensor(x, is_scale: bool):
        if torch.is_tensor(x):
            t = x.to(device=device)
        else:
            t = torch.as_tensor(x, device=device)
        if t.numel() == 1:
            t = t.repeat(B)
        if t.dim() != 1 or t.shape[0] != B:
            raise RuntimeError(
                f"Expected {'scale_factor' if is_scale else 'offset'} of shape (B,), got {tuple(t.shape)}"
            )
        return t

    scale_factor = _to_1d_tensor(scale_factor, is_scale=True).float()
    h_offset = _to_1d_tensor(h_offset, is_scale=False).long()
    w_offset = _to_1d_tensor(w_offset, is_scale=False).long()

    # Prepare intrinsics if provided
    K_is_numpy = False
    if K is not None:
        if torch.is_tensor(K):
            K_t = K.to(device=device, dtype=dtype)
        else:
            K_is_numpy = True
            K_np = np.asarray(K, dtype=np.float32)
            K_t = torch.from_numpy(K_np).to(device=device, dtype=dtype)

        if K_t.dim() == 2:
            K_t = K_t.unsqueeze(0).repeat(B, 1, 1)
        elif K_t.dim() == 3 and K_t.shape[0] == B:
            pass
        else:
            raise RuntimeError(f"Expected K with shape (3,3) or (B,3,3), got {tuple(K_t.shape)}")
    else:
        K_t = None

    imgs_out = []
    Ks_out = [] if K_t is not None else None

    for b in range(B):
        img_b = img[b]  # (C, H, W)

        # Scaling
        s = float(scale_factor[b].item())
        new_h = max(1, int(round(H * s)))
        new_w = max(1, int(round(W * s)))

        img_scaled = F.resize(
            img_b,
            size=[new_h, new_w],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        sx = new_w / W
        sy = new_h / H

        # Crop
        y0 = int(h_offset[b].item())
        x0 = int(w_offset[b].item())
        y1 = min(y0 + crop_height, new_h)
        x1 = min(x0 + crop_width, new_w)

        # Clamp to valid range in case offsets go slightly out of bounds
        y0 = max(0, min(y0, new_h - 1))
        x0 = max(0, min(x0, new_w - 1))
        y1 = max(y0 + 1, y1)
        x1 = max(x0 + 1, x1)

        img_cropped = img_scaled[:, y0:y1, x0:x1]

        # Pad if needed to reach target crop size
        ch, cw = img_cropped.shape[1], img_cropped.shape[2]
        p_left = max(0, int((crop_width - cw) // 2))
        p_right = max(0, crop_width - cw - p_left)
        p_top = max(0, int((crop_height - ch) // 2))
        p_bottom = max(0, crop_height - ch - p_top)

        if p_top > 0 or p_bottom > 0 or p_left > 0 or p_right > 0:
            img_padded = tF.pad(img_cropped, (p_left, p_right, p_top, p_bottom))
        else:
            img_padded = img_cropped

        # Make sure final size is exactly crop_size
        img_padded = img_padded[:, :crop_height, :crop_width]
        imgs_out.append(img_padded)

        if K_t is not None:
            Ks = torch.eye(3, device=device, dtype=dtype)
            Ks[0, 0] = sx
            Ks[1, 1] = sy

            Kc = torch.eye(3, device=device, dtype=dtype)
            Kc[0, 2] = -float(w_offset[b].item())
            Kc[1, 2] = -float(h_offset[b].item())

            Kp = torch.eye(3, device=device, dtype=dtype)
            Kp[0, 2] = float(p_left)
            Kp[1, 2] = float(p_top)

            K_aug_b = Kp @ (Kc @ (Ks @ K_t[b]))
            Ks_out.append(K_aug_b)

    img_aug = torch.stack(imgs_out, dim=0)

    if K_t is None:
        return img_aug

    K_aug_t = torch.stack(Ks_out, dim=0)
    if K_is_numpy:
        return img_aug, K_aug_t.cpu().numpy()
    return img_aug, K_aug_t


def rotate_image(image, camera=None):
  """Rotate image(s) by 90 degrees to convert portrait to landscape and update intrinsics.

  Args:
    image: torch.Tensor or np.ndarray [B, V, C, H, W]
    camera: dict with keys:
      intrinsics: torch.Tensor or np.ndarray [B, V, 3, 3]
      image_size: torch.Tensor or np.ndarray [2]: [H, W]

  Returns:
    image: torch.Tensor or np.ndarray [B, V, C, H, W]
    camera: dict with keys:
      intrinsics: torch.Tensor or np.ndarray [B, V, 3, 3]
      image_size: torch.Tensor or np.ndarray [2]: [H, W]
  """

  # Rotate the image data
  if isinstance(image, torch.Tensor):
    # image shape expected: (..., C, H, W)
    image = torch.rot90(image, k=1, dims=(-2, -1))
  else:
    # Fallback: assume NumPy array with shape (..., C, H, W)
    image = np.rot90(image, k=1, axes=(-2, -1))

  if camera is None:
    return image

  intrinsics = camera["intrinsics"]
  image_size = camera["image_size"]

  # Torch-based camera (batched MATCH representation)
  if isinstance(intrinsics, torch.Tensor):
    device = intrinsics.device
    dtype = intrinsics.dtype

    if isinstance(image_size, torch.Tensor):
      width = image_size[1]
    else:
      width = torch.tensor(image_size[1], dtype=dtype, device=device)

    Rt = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    )
    Rt[1, 2] = width

    camera = dict(camera)
    camera["intrinsics"] = Rt @ intrinsics
    if isinstance(image_size, torch.Tensor):
      camera["image_size"] = torch.flip(image_size, dims=[0])
    else:
      camera["image_size"] = torch.tensor(
          [image_size[1], image_size[0]], dtype=dtype, device=device
      )

    return image, camera

  # NumPy-based camera (TEMPEH-style representation)
  width = image_size[1]
  Rt = np.array(
      [
          [0.0, 1.0, 0.0],
          [-1.0, 0.0, width],
          [0.0, 0.0, 1.0],
      ],
      dtype=intrinsics.dtype,
  )

  camera = dict(camera)
  camera["intrinsics"] = Rt.dot(intrinsics)
  camera["image_size"] = camera["image_size"][::-1]

  return image, camera

class Batch(Dict):
  def squeeze(self):
    '''    
    Removes batch dimension if it is 1
    '''
    squeezed = self.__class__()
    for k, v in self.items():
      if len(v) == 0: 
        pass
      elif len(v) == 1:
        v = v[0]
      else:
        raise ValueError(f'Cannot squeeze key {k} with length {len(v)}')
      squeezed[k] = v
    return squeezed
  
  def unsqueeze(self):
    '''    
    Adds batch dimension if it is missing
    '''
    unsqueezed = self.__class__()
    for k, v in self.items():
      if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
        unsqueezed[k] = v[None]
      elif isinstance(v, dict):
         raise NotImplementedError('Unsqueeze not implemented for nested dicts yet.')
      else:
        unsqueezed[k] = [v]
    return unsqueezed

  def to_device(self, device: torch.device):
    for k, v in self.items():
      if isinstance(v, torch.Tensor):
        self[k] = v.to(device)
    return self

"""
Utility for batch of MATCH data. Provides a set of transformation functions."""
class MatchBatch(Batch):
  IMAGE_KEYS = [
    "image",
    "mask",
    "uv",
    'sg_parts',
  ]
  NEAREST_INTERPOLATION_KEYS = ["uv", 'sg_parts']
  EXTRA_PREFIX = "extra_"
  VIEW_INDEPENDENT_KEYS = ["subject", "sequence", "frame", "verts", "coarse_verts", 'idindex', 'scene_rotation', 'scene_center', 'scene_scale', 'headpose', 'dataset_idx']
  TEMPEH_SCALE = 1000.0  # Geometry scale factor between MATCH and Tempeh


  """
  Batch entries: 
  verts <class 'torch.Tensor'> torch.Size([2, 5741, 3])                       
  coarse_verts <class 'torch.Tensor'> torch.Size([2, 0, 3])                   
  C2W <class 'torch.Tensor'> torch.Size([2, 12, 4, 4])                        
  fxfycxcy <class 'torch.Tensor'> torch.Size([2, 12, 4])                      
  image <class 'torch.Tensor'> torch.Size([2, 12, 3, 393, 256])               
  uv <class 'torch.Tensor'> torch.Size([2, 12, 3, 393, 256])                  
  sg_parts <class 'torch.Tensor'> torch.Size([2, 12, 1, 393, 256])            
  mask <class 'torch.Tensor'> torch.Size([2, 12, 1, 393, 256])                
  headpose <class 'torch.Tensor'> torch.Size([2, 4, 4])                       
  frame <class 'list'> 2                                                      
  camera <class 'torch.Tensor'> torch.Size([2, 12])                           
  subject <class 'list'> 2                                                    
  sequence <class 'list'> 2                                                   
  scene_rotation <class 'torch.Tensor'> torch.Size([2, 3, 3])                 
  scene_center <class 'torch.Tensor'> torch.Size([2, 3])                      
  scene_scale <class 'torch.Tensor'> torch.Size([2])                          
  bboxs <class 'torch.Tensor'> torch.Size([2, 12, 4])                         
  idindex <class 'torch.Tensor'> torch.Size([2])
  (dataset_idx) <class 'torch.Tensor'> torch.Size([2]) [int], only available in multi-dataset

  """

  @property
  def B(self):
    if self.is_batch:
      return self['image'].shape[0]
    else:
      return None

  @property
  def H(self):
    return self['image'].shape[-2]
  
  @property
  def W(self):
    return self['image'].shape[-1]
  
  @property
  def V(self):  # number of views
     return self['image'].shape[-4]
  
  @property
  def is_batch(self):
    return len(self["image"].shape) > 4
  
  @property
  def device(self):
     return self['image'].device

  def resize(self, resolution: int|tuple[int, int]):
    """returns copy of batch with images and intrinsics resized to the desired resolution.

    Args:
      resolution: The desired output height or height and width. If int, width will be adjusted to preserve
        aspect ratio.
    """

    resized_data = dict()
    data = self
    # Resize to the input resolution
    h, w = data["image"].shape[-2:]
    if isinstance(resolution, int):
      target_res_y = resolution
      target_res_x = int(np.round(target_res_y * w / h))
    else:
      target_res_y = resolution[0] 
      target_res_x = resolution[1] 

    if h == target_res_y and w == target_res_x:
      return copy.deepcopy(data)

    nearest_interpolation = [False] * len(self.IMAGE_KEYS)
    for k in self.NEAREST_INTERPOLATION_KEYS:
      nearest_interpolation[self.IMAGE_KEYS.index(k)] = True

    for i in range(len(self.IMAGE_KEYS)):
      key = self.IMAGE_KEYS[i]
      if key in data.keys():
        v = data[key]
        batch_dims = v.shape[:-3]
        v_flattened = torch.flatten(v, end_dim=-4)
        v_resized = tF.interpolate(
            v_flattened,
            size=(target_res_y, target_res_x),
            mode="nearest" if nearest_interpolation[i] else "bilinear",
            align_corners=None if nearest_interpolation[i] else False,
            antialias=False if nearest_interpolation[i] else True,
        )
        v_resized = v_resized.view(batch_dims + v_resized.shape[1:])
        resized_data[key] = v_resized

    # Copy over remaining keys
    for k in set(data.keys()) - set(resized_data.keys()):
      resized_data[k] = data[k]
    return resized_data

  def colorize_bg(self, color: list[float]):
    """Colorizes the background (applies color both to bg and image).

    Args:
      data: A dictionary of data with keys:
        image: ((B), V, 3, H, W) 0...1
        bg: ((B), V, 3, H, W) 0...1
        mask: ((B),V, 1, H, W) 0...1
    """

    colorized_data = self.__class__()
    data = self
    mask = data["mask"]
    batch_dims = mask.shape[:-3]
    color = torch.tensor(color, device=mask.device, dtype=mask.dtype).reshape(
        *([1] * len(batch_dims) + [-1, 1, 1])
    )
    for k, v in data.items():
      if k in ["image"]:
        v = v * mask + (1 - mask) * color
      elif k in ['bg']:
        v = torch.ones_like(mask) * color
      colorized_data[k] = v
    return colorized_data

  def index_select_views(self, idcs: torch.Tensor):
    data = self
    out_dict = self.__class__()
    for k, v in data.items():
      if (k not in self.VIEW_INDEPENDENT_KEYS) and (not k.startswith(self.EXTRA_PREFIX)):
        v = torch.index_select(v, dim=1 if self.is_batch else 0, index=idcs)
      out_dict[k] = v
    return out_dict
  
  def subsample_views(
    self,
    nviews: int | None = None,
    ninputviews: int | None = None,
    random: bool = False,
  ) -> dict[str, torch.Tensor]:
    """Randomly selects a subset of the views from the multi-view data.

    Selects a subset of views from the augmented images and augmented camera
    parameters. The original images and camera intrinsics in the data dict remain
    unchanged.

    if random:
      input views are sampled randomly
    else:
      input views (first ninputviews) are sampled evenly (deterministic)from the
      beginning of the
      view sequence and ensures that the target views lie between them

    Args:
      data: Multi-view images and camera data. Can be a single sample or a batch.
      nviews: Number of views to select.
      ninputviews: Number of input views to select.
      deterministic: Whether to use a deterministic view selection.

    Returns:
      A dict with a subset of views for the augmented images and cameras.
    """
    if nviews is None:
      nviews = self.V
    if ninputviews is None:
      ninputviews = nviews
    
    if random:
      view_samples = torch.randperm(self.V)[:nviews]
    else:
      view_idcs = torch.round(torch.linspace(0, self.V - 1, nviews)).long()
      input_view_idcs = torch.gather(
          view_idcs,
          0,
          torch.round(torch.linspace(0, nviews - 1, ninputviews)).long(),
      )
      noninput_view_idcs = torch.sort(
          torch.tensor(
              list(set(view_idcs.tolist()) - set(input_view_idcs.tolist()))
          )
      ).values.long()
      view_samples = torch.cat([input_view_idcs, noninput_view_idcs], dim=0)

    return self.index_select_views(view_samples)


  
  def cat_extra_keys(self):
    '''
    moves all keys with prefix 'extra_' to their unprefixed version (e.g. 'extra_image' -> 'image')
    '''
    is_batch = self.is_batch
    if is_batch:
      batch = self
    else:
      batch = self.unsqueeze()
    
    out_batch = self.__class__()
    keys = list(batch.keys())
    for k in keys:
      if k.startswith('extra_'):
        continue
      else:
        extra_k = 'extra_'+k
        if extra_k in batch:
          out_batch[k] = torch.cat([batch[k], batch[extra_k]], dim=1)
        else:
          out_batch[k] = batch[k]
    
    if not is_batch:
       out_batch = out_batch.squeeze()
    return out_batch

  def get_subdirpaths(self)->List[str]:
    '''   
    returns string like "{subject}/{sequence}/{frame}" for each item in the batch
    '''
    is_batch = self.is_batch
    if not is_batch:
      batch = self.unsqueeze()
    else:
      batch = self
       
    b = len(batch['verts'])
    subdirpaths = list()
    for i in range(b):
      subject = batch['subject'][i]
      sequence = batch['sequence'][i]
      frame = batch['frame'][i]
      subdirpaths.append(f'{subject}/{sequence}/{frame}')
    
    if not is_batch:
      subdirpaths = subdirpaths[0]
    return subdirpaths
  
  def to_dict(self):
     return dict(self)

class TempehBatch(Batch):
  """
  Utility for batch of Tempeh data.
  """
  IMG_MEAN = [0.5] * 3
  IMG_STD = [0.5] * 3

  @classmethod
  def from_match_batch(cls, match_batch: MatchBatch, scale_min=0.9, scale_max=1.1, brightness_sigma=0.33):
    batch = match_batch.to_dict()
    B, V, C, H, W = batch['image'].shape
    device = match_batch.device

    image = batch['image']
    mask = batch['mask']

    # black bg
    image = image * mask

    c2w = batch['C2W']
    w2c = geo_util.invert_c2w(c2w)
    K = torch.eye(3, dtype=torch.float32, device=device).repeat(B, V, 1, 1)
    K[:, :, 0, 0] = batch['fxfycxcy'][:, :, 0] * W
    K[:, :, 1, 1] = batch['fxfycxcy'][:, :, 1] * H
    K[:, :, 0, 2] = batch['fxfycxcy'][:, :, 2] * W - 0.5
    K[:, :, 1, 2] = batch['fxfycxcy'][:, :, 3] * H - 0.5
    
    camera = dict(
      intrinsics = K,
      extrinsics = w2c[:, :, :3, :],
      camera_center = c2w[:, :, :3, -1],
      view_direction = c2w[:, :, :3, -2],
      image_size=torch.tensor([H, W], device=device),  # h, w
      name = batch['camera'],
      radial_distortion = torch.zeros(B, V, 2, dtype=torch.float32, device=device)
    )

    # geometric augmentation by random scaling and cropping
    rng = np.random.default_rng()
    crop_size = (H, W)
    scale_factor = torch.tensor(scale_min + (scale_max - scale_min) * rng.random((B * V)), device=device)
    h_offset, w_offset = torch.zeros(B * V, dtype=torch.int32, device=device), torch.zeros(B * V, dtype=torch.int32, device=device)
    image_augmented, intrinsics_augmented = scale_crop(
      einops.rearrange(image, 'b v c h w -> (b v) c h w'), 
      crop_size, 
      h_offset, 
      w_offset, 
      scale_factor, 
      K=einops.rearrange(camera['intrinsics'], 'b v c1 c2 -> (b v) c1 c2'))
    image_augmented = einops.rearrange(image_augmented, '(b v) c h w -> b v c h w', b=B, v=V)
    intrinsics_augmented = einops.rearrange(intrinsics_augmented, '(b v) c1 c2 -> b v c1 c2', b=B, v=V)

    # random brightness perturbation
    perturb = 1.0 + brightness_sigma * torch.tensor(rng.standard_normal((B, V, 3, 1,1)), device=device, dtype=torch.float32)
    image_augmented = image_augmented * perturb
    image_augmented = torch.clamp(image_augmented, 0., 1.)

    # normalize rgb
    image = normalize_image(image, mean=cls.IMG_MEAN, std=cls.IMG_STD)
    image_augmented = normalize_image(image_augmented, mean=cls.IMG_MEAN, std=cls.IMG_STD)

    data = {
      # img
      'color_images': [],
      'stereo_images': image,
      'color_images_augmented': [],
      'stereo_images_augmented': image_augmented,

      # camera
      'color_camera_intrinsics': [],
      'color_camera_extrinsics': [],
      'color_camera_distortions': [],
      'color_camera_centers': [],
      'stereo_camera_intrinsics': camera['intrinsics'],
      'stereo_camera_extrinsics': camera['extrinsics'],
      'stereo_camera_distortions': camera['radial_distortion'],
      'stereo_camera_centers': camera['camera_center'],
      
      'color_camera_intrinsics_augmented': [],
      'stereo_camera_intrinsics_augmented': intrinsics_augmented,

      # meta
      # 'index': index,
      'subject': batch['subject'],
      'sequence': batch['sequence'],
      'frame': batch['frame'],  
      'camera': batch['camera'],

      # meshes
      'v_scan': batch['verts'],
      'v_registration': batch['verts'],
      'f_registration': torch.from_numpy(AVA_TEMPLATE_MESH_INFO['vi'].astype(np.int64)).to(device).repeat(B, 1, 1),
      'v_reg_sampled': batch['verts'],
      'f_reg_sampled': torch.from_numpy(AVA_TEMPLATE_MESH_INFO['vi'].astype(np.int64)).to(device).repeat(B, 1, 1)
    }

    # from m -> mm
    assert len(data['color_camera_centers']) == 0, 'Conversion not implemented for color cameras'
    data['v_scan'] = data['v_scan'] * 1000.0  
    data['v_registration'] = data['v_registration'] * 1000.0
    data['v_reg_sampled'] = data['v_reg_sampled'] * 1000.0 
    if 'v_reg_global' in data:
      data['v_reg_global'] = data['v_reg_global'] * 1000.0  
    data['stereo_camera_centers'] = data['stereo_camera_centers']
    data['stereo_camera_extrinsics'][..., :3, :3] = data['stereo_camera_extrinsics'][..., :3, :3] / 1000  # extrinsics go from mm to m


    if 'dataset_idx' in batch:
      data['dataset_idx'] = batch['dataset_idx']
    return cls(data)

  @property
  def V(self):  # number of views
     return self['images'].shape[-4]



def batch_extra_keys_to_main_keys(batch: dict):
  '''
  moves all keys with prefix 'extra_' to their unprefixed version (e.g. 'extra_image' -> 'image')
  '''
  out_batch = dict()
  keys = list(batch.keys())
  for k in keys:
    if k.startswith('extra_'):
      continue
    else:
      extra_k = 'extra_'+k
      if extra_k in batch:
        out_batch[k] = batch[extra_k]
      else:
        out_batch[k] = batch[k]
  return out_batch



def batch_to_samples(batch):
  samples = list()
  for k, v in batch.items():
    if isinstance(v, dict):
      v = batch_to_samples(v)
    for i in range(len(v)):
        if len(samples)<=i:
          samples.append(dict())
        samples[i][k] = v[i]

  return samples

def headcrop_gtempeh_batch(batch, head_crop_width:int, head_crop_height:int):
  crop_w_in = crop_w_out = head_crop_width
  crop_h_in = crop_h_out = head_crop_height
  head_center_world = batch['headpose'][:, :, -1]  # (B, 4,)
  W2C = geo_util.invert_c2w(batch['C2W'])
  B, V, C, H, W = batch['image'].shape
  K = torch.zeros((B, V, 3, 3), device=W2C.device)
  K[..., -1, -1] = 1
  K[..., 0, 0] = batch['fxfycxcy'][..., 0] * W
  K[..., 1, 1] = batch['fxfycxcy'][..., 1] * H
  K[..., 0, 2] = batch['fxfycxcy'][..., 2] * W
  K[..., 1, 2] = batch['fxfycxcy'][..., 3] * H
  head_center_cam = einops.einsum(W2C, head_center_world, 'B Nviews i j, B j -> B Nviews i')[:,:, :3]  # (B, Nviews, 3)
  head_center_screen = einops.einsum(K, head_center_cam, 'B Nviews i j, B Nviews j -> B Nviews i')  # (B, Nviews, 3)
  head_center_screen[:,:,  :2] = head_center_screen[:, :,  :2] / head_center_screen[:,:,  2:]
  crop_center = head_center_screen[:,:, :2]

  crop_t = crop_center[:, :, 1] - crop_h_in / 2
  crop_l = crop_center[:, :, 0] - crop_w_in / 2

  # making sure entire head crop is on image, if not, move such that it is
  crop_t = torch.maximum(torch.minimum(crop_t, torch.ones_like(crop_t) * (H-crop_h_in)), torch.zeros_like(crop_t))
  crop_l = torch.maximum(torch.minimum(crop_l, torch.ones_like(crop_l) * (W-crop_w_in)), torch.zeros_like(crop_l))
  crop_t = crop_t.to(torch.int32)  # (B, V,)
  crop_l = crop_l.to(torch.int32)  # (B, V,)
  if torch.any(crop_l + crop_w_in > W):
      crop_h_in = int(torch.min(crop_h_in * (W-crop_l) / crop_w_in))
      crop_w_in = int(torch.min(W-crop_l))
  if torch.any(crop_t + crop_h_in > H):
      crop_w_in = int(torch.min(crop_w_in * (H-crop_t) / crop_h_in))
      crop_h_in = int(torch.min(H-crop_t))
  
  K[:, :, 0, 2] = K[:, :, 0, 2] - crop_l
  K[:, :, 1, 2] = K[:, :, 1, 2] - crop_t
  K[:,:, 0] =  K[:,:, 0] * crop_w_out / crop_w_in
  K[:,:, 1] =  K[:,:, 1] * crop_h_out / crop_h_in

  cropped_batch = dict()
  for k, v in batch.items():
      if k in ['image', 'bg', 'mask', 'uv', 'sg_parts']:
          v = general_util.crop_tensor(einops.rearrange(v, 'b v c h w -> (b v) c h w'), crop_t = torch.flatten(crop_t), crop_l=torch.flatten(crop_l), crop_height=crop_h_in, crop_width=crop_w_in)
          if (crop_h_in != crop_h_out) or (crop_w_in != crop_w_out):
              v = F.resize(v, (crop_h_out, crop_w_out), interpolation=InterpolationMode.NEAREST if k in ['uv', 'sg_parts'] else InterpolationMode.BILINEAR)
          v = einops.rearrange(v, '(b v) c h w -> b v c h w', b=B, v=V)
      elif k == 'sapiens_features': 
          raise NotImplementedError('Batch cropping not implemented for sapiens features.')
      elif k == 'fxfycxcy':
          v = torch.stack([K[:, :, 0,0]/crop_w_out, K[:, :, 1,1]/crop_h_out, K[:, :, 0,2]/crop_w_out, K[:, :, 1,2]/crop_h_out], dim=-1)
      cropped_batch[k] = v
  return cropped_batch


def normalize_image(image, mean, std):
  '''
  Args:
    image: torch.Tensor with shape (B,C,H,W)
    mean: list of floats with shape (C,)
    std: list of floats with shape (C,)
  Returns:
    image: torch.Tensor with shape (B,C,H,W)
  '''
  mean = torch.tensor(mean).to(image)
  std = torch.tensor(std).to(image)
  return ( image - mean.view(1, -1, 1, 1) ) / std.view(1, -1, 1, 1)

def random_color_augment_images(img: torch.Tensor, brightness_range:float, contrast_range:float, saturation_range:float, hue_range:float, p_grayscale:float) -> torch.Tensor:
    """
    Apply the same random color-based augmentations to a batch of images.
    
    Args:
        img: Tensor of shape (B, 3, H, W), values in [0, 1].
    
    Returns:
        Augmented images of same shape.
    """
    assert img.dim() == 4 and img.size(1) == 3, "Shape must be (B, 3, H, W)"
    B, C, H, W = img.shape

    # Sample random parameters once
    brightness_factor = random.uniform(1.-brightness_range, 1.+brightness_range)   # 20% jitter
    contrast_factor   = random.uniform(1.-contrast_range, 1.+contrast_range)
    saturation_factor = random.uniform(1.-saturation_range, 1.+saturation_range)
    hue_factor        = random.uniform(-hue_range, hue_range)  # hue shift
    to_grayscale      = random.random() < p_grayscale      # 20% chance

    img = einops.rearrange(img, 'b c h w -> c (b h) w')
    # torchvision expects (C, H, W), so this is fine
    if brightness_range!=0:
      img = F.adjust_brightness(img, brightness_factor)
    if contrast_range != 0:
      img = F.adjust_contrast(img, contrast_factor)
    if saturation_range != 0:
      img = F.adjust_saturation(img, saturation_factor)
    if hue_range != 0:
      img = F.adjust_hue(img, hue_factor)
    if to_grayscale:
        img = F.rgb_to_grayscale(img, num_output_channels=3)
    img = einops.rearrange(img, 'c (b h) w -> b c h w', b=B)
    img = img.clip(0,1)

    return img


class RepeatFewBatchDataLoader(IterableDataset):
    def __init__(self, base_dataloader, num_cached_batches=-1):
        super().__init__()
        if num_cached_batches == -1:
           num_cached_batches = len(base_dataloader)
        self.base_dataloader = base_dataloader
        self.num_cached_batches = num_cached_batches
        self._cache = []
        self._base_iterator = None
        self.dataset = base_dataloader.dataset
        self.batch_size = getattr(base_dataloader, 'batch_size', None)


    def __iter__(self):
        # Reinitialize base iterator each time __iter__ is called
        if self._base_iterator is None:
            self._base_iterator = iter(self.base_dataloader)

        idx = 0
        while True:
            if idx < len(self._cache):
                # Use cached batch
                yield self._cache[idx]
            elif len(self._cache) < self.num_cached_batches:
                # Cache a new batch
                try:
                    batch = next(self._base_iterator)
                    self._cache.append(batch)
                    yield batch
                except StopIteration:
                    # No more batches to cache — switch to cycling over what we have
                    if not self._cache:
                        raise ValueError("Base dataloader exhausted before any batches could be cached.")
                    idx = 0
                    continue
            else:
                # All batches cached — loop forever
                idx = 0
                continue

            idx += 1


def cycle(iterable):
    """infinite looping over an iterable"""
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class SkippableError(Exception):
    def __init__(self, message=""):
        super().__init__(message)
        self.message = message

def skippable_zpath(path:file_util.Path, content:str):
  try:
    return ZipPath(path, content)
  except: FileNotFoundError


def rigid_transform_gtempeh_sample(sample: dict, trafo: torch.Tensor):
  """ Rigidly transforms a gtempeh sample
  
  Args: 
    sample: gtempeh sample
    trafo: (4,4) rigid transformation to apply
  """

  transformed_sample = dict()
  rotation = trafo[:3, :3]
  translation = trafo[:3, -1]
  for k, v in sample.items():
    if k in ['verts', 'stage1verts']:
      v = (rotation @ v.T).T + translation[None]
    elif k in ['C2W']:
      v = einops.einsum(trafo, v, 'i j, v j k -> v i k')
    elif k in ['headpose']:
      v = trafo @ v

    # TODO: implement for 'scene_rotation', 'scene_center', and 'scene_scale' keys    
    
    transformed_sample[k] = v

  return transformed_sample


def resize_sample_tf(
    data: dict[str, tf.Tensor],
    resize_keys: list[str],
    nearest_interpolation_keys: list[str] | None = None,
    resolution: int | None = None,
) -> dict[str, tf.Tensor]:
  """Resizes the images in the data dict to the desired resolution."""
  if resolution is None:
    return data
  h_old = tf.shape(data["image"])[-3]
  w_old = tf.shape(data["image"])[-2]
  if h_old == resolution:
    return data

  return_dict = dict()
  h_new = resolution
  w_new = tf.cast(tf.round(h_new * w_old / h_old), dtype=tf.int32)
  if nearest_interpolation_keys is None:
    nearest_interpolation_keys = []
  for k, v in data.items():
    if k in resize_keys:
      v = tf.image.resize(
          v,
          size=(h_new, w_new),
          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
          if k in nearest_interpolation_keys
          else tf.image.ResizeMethod.BILINEAR,
          antialias=False if k in nearest_interpolation_keys else True,
      )

    return_dict[k] = v
  return return_dict


def random_view_selection_tf(
    data: dict[str, tf.Tensor],
    nviews: int | None = None,
    ninputviews: int | None = None,
    even_sampling: bool = False,
) -> dict[str, tf.Tensor]:
  """Randomly selects a subset of the views from the multi-view data.

  Selects a subset of views from the augmented images and augmented camera
  parameters. The original images and camera intrinsics in the data dict remain
  unchanged.

  if even_sampling:
    input views (first ninputviews) are sampled evenly (deterministic) from the
    beginning of the
    view sequence and ensures that the target views lie between them
  else:
    input views are sampled randomly

  Args:
    data: Multi-view images and camera data.

  Returns:
    A dict with a subset of views for the augmented images and cameras.
  """
  if nviews is None or ninputviews is None:
    return data
  else:
    nviews: int
    ninputviews: int
    V = tf.shape(data["image"])[0]
    if (
        even_sampling
    ):  # deterministic view selection for eval, evenly distributed input views at the beginning of the view sequence
      view_idcs = tf.cast(
          tf.round(tf.linspace(0, V - 1, nviews)), dtype=tf.int32
      )
      input_view_idcs = tf.gather(
          view_idcs,
          tf.cast(
              tf.round(tf.linspace(0, nviews - 1, ninputviews)), dtype=tf.int32
          ),
      )
      noninput_view_idcs = tf.sort(
          tf.sets.difference(
              tf.expand_dims(view_idcs, axis=0),
              tf.expand_dims(input_view_idcs, axis=0),
          ).values
      )
      view_samples = tf.concat([input_view_idcs, noninput_view_idcs], axis=0)

    else:
      view_samples = tf.random.shuffle(tf.range(V))
      view_samples = tf.slice(view_samples, begin=[0], size=[nviews])
    out_dict = dict()
    for k, v in data.items():
      if k not in [
          "subject",
          "sequence",
          "frame",
          "verts",
          "stage1verts",
          "dataset_idx",
      ]:
        v = tf.gather(v, view_samples, axis=0)
      out_dict[k] = v
    return out_dict


def visualize_camera_grid(samples: Dict[str,dict], outpath: str):
    """
    Visualizes a sample's camera grid:
    - 2D grid of images with camera IDs (saved as PNG)
    - 3D visualization of camera extrinsics and mesh (saved as HTML)

    Args:
        sample: named samples. keys are sample names, values are dictionary with keys:
            - image: (V, 3, H, W) tensor
            - C2W: (V, 4, 4) tensor
            - verts: (N, 3) tensor
            - camera: list of camera IDs
        outpath: Output file path (without extension).
                 Will save {outpath}_grid.png and {outpath}_3d.html
    """

    fig_3d = go.Figure()
    axis_length = 0.1  # length of axis cones
    mesh_vis_stride = 1

    
    for sample_name, sample in samples.items():
      images = sample["image"]      # (V, 3, H, W)
      C2W = sample["C2W"]           # (V, 4, 4)
      verts = sample["verts"]       # (N, 3)
      camera_ids = list(map(str, sample["camera"].tolist())) # list of length V
      if 'extra_image' in sample:
        extra_images = sample["extra_image"]      # (V, 3, H, W)
        extra_C2W = sample["extra_C2W"]           # (V, 4, 4)
        extra_camera_ids = list(map(str, sample["extra_camera"].tolist())) # list of length V         
        all_images = torch.cat([images, extra_images], dim=0)
        all_C2W = torch.cat([C2W, extra_C2W], dim=0)
        all_camera_ids = camera_ids + ['extra_'+cid for cid in extra_camera_ids]
      else:
        all_images = images
        all_C2W = C2W
        all_camera_ids = camera_ids

      # Ensure output directory exists
      file_util.Path(outpath).parent.mkdir(parents=True, exist_ok=True)

      V = all_images.shape[0]
      grid_size = math.ceil(math.sqrt(V))

      # --- 2D image grid ---
      fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
      axes = axes.flatten()

      for i in range(grid_size * grid_size):
          ax = axes[i]
          ax.axis('off')
          if i < V:
              img = all_images[i].permute(1, 2, 0).cpu().numpy()
              ax.imshow(img)
              ax.set_title(str(all_camera_ids[i]))

      plt.tight_layout()
      plt.savefig(f"{outpath}_grid_{sample_name}.jpg")
      plt.close(fig)
      print(f'Saved 2D grid to {outpath}_grid_{sample_name}.jpg')

      # --- 3D visualization ---
      # Add mesh scatter
      verts_np = verts.cpu().numpy()
      verts_sampled = verts_np[::mesh_vis_stride]
      fig_3d.add_trace(go.Scatter3d(
          x=verts_sampled[:,0], y=verts_sampled[:,1], z=verts_sampled[:,2],
          mode='markers',
          marker=dict(size=2),
          name=f'Mesh_{sample_name}'
      ))

      # For each camera
      for i in range(V):
          c2w = all_C2W[i].cpu().numpy()
          origin = c2w[:3, 3]

          x, y, z = origin      

          el = np.rad2deg(np.arctan2(-y, (x**2 + z**2)**.5))
          az = np.rad2deg(np.arctan2(x, -z))

          # Local axes directions
          x_dir = c2w[:3, 0]
          y_dir = c2w[:3, 1]
          z_dir = c2w[:3, 2]

          # Draw cones for each axis
          for dir_vec, color, name in zip([x_dir, y_dir, z_dir], ['red', 'green', 'blue'], ['X', 'Y', 'Z']):
              tip = origin + dir_vec * axis_length

              fig_3d.add_trace(go.Cone(
                  x=[origin[0]], y=[origin[1]], z=[origin[2]],
                  u=[tip[0] - origin[0]], v=[tip[1] - origin[1]], w=[tip[2] - origin[2]],
                  anchor="tail",
                  showscale=False,
                  colorscale=[[0, color], [1, color]],
                  sizemode="absolute",
                  sizeref=axis_length / 5,
                  name=f'Cam{all_camera_ids[i]}_{sample_name}_{name}'
              ))

          # Add floating text with camera id
          fig_3d.add_trace(go.Scatter3d(
              x=[origin[0]], y=[origin[1]], z=[origin[2]],
              mode='text',
              text=[f"{all_camera_ids[i]}_{sample_name}_az{int(az)}_el{int(el)}"],
              # text=[f"{camera_ids[i]}_{sample_name}"],
              textposition="top center",
              textfont=dict(color='black', size=12),
              showlegend=False
          ))

    fig_3d.update_layout(
        scene=dict(aspectmode='data'),
        title='Camera Extrinsics and Mesh'
    )

    fig_3d.write_html(f"{outpath}_3d.html")

    print(f"Saved 3D plot to {outpath}_3d.html")



def filter_cameras_by_angle_range(camera_positions: np.ndarray, angle_range: list, center=np.array([0, 0, 0.])):
    """
       Assumes when looking from face's position: x: left, y: down, -z: lookat (adopting ava256 convention)

    Args:
        camera_positions: (N, 3)
        sample_angles: list of target angles in degrees.
            - [az, el]: fixed target angle
            - [[az_min, az_max], [el_min, el_max]]: sample uniformly in range
        center: center point cameras look at
        unique: enforce unique cameras

    Returns:
        indices: list of selected indices into camera_positions
    """
    camera_positions = camera_positions - center
    x, y, z = camera_positions[..., 0], camera_positions[..., 1], camera_positions[..., 2]
    el = np.rad2deg(np.arctan2(-y, (x**2 + z**2)**.5))
    az = np.rad2deg(np.arctan2(x, -z))
    az_min, az_max = angle_range[0]
    el_min, el_max = angle_range[1]

    mask = (el_min <= el) & (el <= el_max) & (az_min <= az) & (az <= az_max)
    return np.where(mask)[0].tolist()
         
   

def sample_cameras_from_angles(camera_positions: np.ndarray, sample_angles: list, center=np.array([0, 0, 0.]), unique=True, ncams_per_angle=1, temperature=1):
    """ Samples closest cameras matching target angles (az, el).

    Assumes when looking from face's position: x: left, y: down, -z: lookat (adopting ava256 convention)

    Args:
        camera_positions: (N, 3)
        sample_angles: list of target angles in degrees.
            - [az, el]: fixed target angle
            - [[az_min, az_max], [el_min, el_max]]: sample uniformly in range
        center: center point cameras look at
        unique: enforce unique cameras

    Returns:
        indices: list of selected indices into camera_positions
    """
    camera_dirs = camera_positions - center
    camera_dirs /= np.linalg.norm(camera_dirs, axis=1, keepdims=True)  # (N,3)

    N = camera_positions.shape[0]
    indices = []
    used = np.zeros(N, dtype=bool)

    for angle_spec in sample_angles:
        angle_cam_idcs = list()
        # Sample azimuth and elevation in degrees
        if isinstance(angle_spec[0], list):
            az = np.random.uniform(*angle_spec[0])
        else:
            az = angle_spec[0]
        if isinstance(angle_spec[1], list):
            el = np.random.uniform(*angle_spec[1])
        else:
            el = angle_spec[1]

        # Convert to radians
        az_rad, el_rad = np.deg2rad(az), np.deg2rad(el)

        # Convert to Cartesian: shape (3,)
        target_dir = np.array([
            np.sin(az_rad) * np.cos(el_rad),
            - np.sin(el_rad),
            - np.cos(az_rad) * np.cos(el_rad),
        ])

        # Vectorized cosine similarity
        sims = camera_dirs @ target_dir  # shape (N,)

        # Find best index (highest similarity)
        sorted_idx = np.argsort(-sims)
        if unique:
            # take first unused
            counter = 0
            for idx in sorted_idx:
                if not used[idx]:
                    angle_cam_idcs.append(idx)
                    used[idx] = True
                    counter += 1
                    if counter == max(ncams_per_angle, temperature):
                      break
        else:
            angle_cam_idcs = sorted_idx[:max(ncams_per_angle, temperature)].tolist()
        if temperature>1:
           random.shuffle(angle_cam_idcs)
        angle_cam_idcs = angle_cam_idcs[:ncams_per_angle]
        indices.extend(angle_cam_idcs)

    return indices


def path_2_zippath(path:str):
   zipfile_path, content_path = path.split('.zip/')
   return ZipPath(zipfile_path + '.zip', content_path)

def visualize_sample(
    sample: dict, outpath: str, faces: np.ndarray|None = None, s=6, show=False
):
  """Visualizes a sample.

  Args:
    sample: A dictionary of data.
    faces: A numpy array of faces. (F, 3)
    outpath: The output image path.
    s: The size of the figure.
    show: Whether to show the figure.
  """
  if show:
    plt.switch_backend("TkAgg")
  # visualize sample
  image = sample["image"]  # (V, 3, H, W), 0-1
  mask = sample["mask"]  # (V, 1, H, W), 0-1
  C2W = sample["C2W"]  # (V, 4, 4),
  W2C = torch.linalg.inv(C2W)  # (V, 4, 4)
  # normal = sample["normal"]  # (V, 3, H, W), 0-1
  uv = sample["uv"]  # (V, 3, H, W), 0-1
  gt_verts = sample["verts"]  # (N, 3)
  coarse_verts = sample["coarse_verts"]  # (N, 3)
  fxfycxcy = sample["fxfycxcy"]  # (V, 4)
  V, _, H, W = image.shape

  gt_verts_screen = geo_util.project_points_to_screen(gt_verts[None], c2w=C2W[None], fxfycxcy=fxfycxcy[None], H=H, W=W)[0]
  coarse_verts_screen = geo_util.project_points_to_screen(coarse_verts[None], c2w=C2W[None], fxfycxcy=fxfycxcy[None], H=H, W=W)[0]

  fig, axes = plt.subplots(
      8, V, figsize=(V * s, 8 * s), squeeze=False
  )
  for v in range(V):
    axes[0, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[0, v].set_title(f"Image {v}")
    axes[0, v].axis("off")
    axes[2, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[2, v].scatter(
        gt_verts_screen[v, :, 0]
        - 0.5,  # correcting for matplotlib assumes top left pixel center to have coords 0.0, 0.0 but visual_sfm assumes 0.5, 0.5
        gt_verts_screen[v, :, 1] - 0.5,
        color="blue",
        s=0.5,
    )
    axes[2, v].set_title(f"gt_mesh_overlay {v}")
    axes[2, v].axis("off")
    axes[3, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[3, v].scatter(
        coarse_verts_screen[v, :, 0]
        - 0.5,  # correcting for matplotlib assumes top left pixel center to have coords 0.0, 0.0 but visual_sfm assumes 0.5, 0.5
        coarse_verts_screen[v, :, 1] - 0.5,
        color="red",
        s=0.5,
    )
    axes[3, v].set_title(f"coarse_mesh_overlay {v}")
    axes[3, v].axis("off")
    axes[4, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[4, v].imshow(mask[v].numpy().squeeze(), cmap="gray", alpha=0.7)
    axes[4, v].set_title(f"Mask {v}")
    axes[4, v].axis("off")
    axes[5, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[5, v].imshow(sample['sg_parts'][v].numpy().transpose(1, 2, 0)[..., 0], cmap='jet', alpha=0.7)
    axes[5, v].set_title(f"Segmentation {v}")
    axes[5, v].axis("off")
    axes[6, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[6, v].imshow(uv[v].numpy().transpose(1, 2, 0), alpha=0.7)
    axes[6, v].set_title(f"UV Map {v}")
    axes[6, v].axis("off")

  plt.tight_layout()
  if show:
    plt.show()

  output_filename = outpath
  file_util.Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(output_filename)
  print(f"Wrote sample to {output_filename}")
  plt.close(fig)

  if faces is not None:
    # plot mesh
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=gt_verts[:, 0],
                y=gt_verts[:, 1],
                z=gt_verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="gt_mesh",
            ),
              go.Mesh3d(
                x=coarse_verts[:, 0],
                y=coarse_verts[:, 1],
                z=coarse_verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name="stage1_mesh",
            )
        ]
    )
    fig.update_layout(
        title="Mesh",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    output_filename = outpath + "mesh.html"
    pio.write_html(fig, file=output_filename)
    print(f"Wrote mesh to {output_filename}")
  
  visualize_camera_grid(dict(sample=sample), outpath)



