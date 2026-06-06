import os
import einops
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import pudb
from gtempeh_utils import file_helper
import tensorflow as tf
import torch
from gtempeh_utils import geo_util, mesh_util
import math
from zipp import Path as ZipPath
from typing import List
from torch.utils.data import IterableDataset
import tqdm
from typing import *
import torch.nn.functional as tF

# TEMPEH stuff
from utils.camera import load_mpi_camera, rotate_image
from utils.data_augment import get_random_crop_offsets, scale_crop
import copy

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

def skippable_zpath(path:file_helper.Path, content:str):
  try:
    return ZipPath(path, content)
  except: FileNotFoundError

def visualize_sample_wojciech(
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    image: torch.Tensor,
    verts: torch.Tensor,
    s=6.0,
    outfile: Optional[str] = None,
):
  """Args:

  extrinsics: (V, 4, 4)
  intrinsics: (V, 4)
  image: (V, 3, H, W)
  verts: (N, 3)
  s: scale factor for the figure
  """

  fx = intrinsics[:, 0, 0]
  fy = intrinsics[:, 1, 1]
  cx = intrinsics[:, 0, 2]
  cy = intrinsics[:, 1, 2]
  verts = torch.cat([verts, torch.ones_like(verts[..., :1])], dim=-1)
  verts_cam = einops.einsum(extrinsics, verts, "v i j, n j -> v n i")
  vert_x_2d = fx[:, None] * verts_cam[..., 0] / verts_cam[..., 2] + cx[:, None]
  vert_y_2d = fy[:, None] * verts_cam[..., 1] / verts_cam[..., 2] + cy[:, None]
  verts_screen = (
      torch.stack([vert_x_2d, vert_y_2d], dim=-1).cpu().numpy()
  )  # (V, N, 2)

  V, _, H, W = image.shape
  fig, axes = plt.subplots(2, V, figsize=(V * s, 2 * s), squeeze=False)
  for v in range(V):
    axes[0, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[0, v].set_title(f"Image {v}")
    axes[0, v].axis("off")
    axes[1, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[1, v].scatter(
        verts_screen[v, :, 0]
        - 0.5,  # correcting for matplotlib assumes top left pixel center to have coords 0.0, 0.0 but visual_sfm assumes 0.5, 0.5
        verts_screen[v, :, 1] - 0.5,
        color="blue",
        s=0.5,
    )
    axes[1, v].set_title(f"mesh_overlay {v}")
    axes[1, v].axis("off")

  plt.tight_layout()
  if outfile:
    file_helper.Path(outfile).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(outfile)
  plt.show()
  plt.close(fig)


def get_subdirpaths_from_batch(batch)->List[str]:
  '''
  
  Args:
    batch: gtempeh batch
  '''
  b = len(batch['verts'])
  subdirpaths = list()
  for i in range(b):
    subject = batch['subject'][i]
    sequence = batch['sequence'][i]
    frame = batch['frame'][i]
    subdirpaths.append(f'{subject}/{sequence}/{frame}')
  return subdirpaths

def tempeh_normalize_image(image, mean=np.array([0.485, 0.456, 0.406], dtype=np.float32), std=np.array([0.229, 0.224, 0.225], dtype=np.float32)):
  # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
  if image.ndim !=3 or image.shape[2] != 3:
      raise RuntimeError(f'invalid image shape {image.shape}')
  else:
      return ( image - mean.reshape((1,1,3)) ) / std.reshape((1,1,3))


def tempeh_unnormalize_image(image, mean=np.array([0.485, 0.456, 0.406], dtype=np.float32), std=np.array([0.229, 0.224, 0.225], dtype=np.float32)):
  # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
  if image.ndim !=3 or image.shape[2] != 3:
      raise RuntimeError(f'invalid image shape {image.shape}')
  else:
      return image *  std.reshape((1,1,3)) + mean.reshape((1,1,3))


ava_template_mesh_info = mesh_util.load_obj(str("/home/mprinzler/projects/gintern/gtempeh/assets/ava256/face_topology_cleaned.obj"))


GTEMPEH_IMAGE_KEYS = [
    "image",
    "bg",
    "mask",
    "albedo",
    "normal",
    "depth",
    "mr",
    "canny",
    "uv",
    'sg_parts',
]

def resize_gtempeh_sample(data_chunk: dict, resolution: Union[int,Tuple[int, int]]):
  """resizes the data chunk to the desired resolution.

  Args:
    data_chunk: A data chunk from the holobooth dataset. values of data chunk
      that are supposed to be resized are expected to be of shape (B, V, C, H,
      W)
    resolution: The desired output height or height and width. If int, width will be adjusted to preserve
      aspect ratio.
  """

  resized_data_chunk = dict()

  # Resize to the input resolution
  h, w = data_chunk["image"].shape[-2:]
  if isinstance(resolution, int):
    target_res_y = resolution
    target_res_x = int(np.round(target_res_y * w / h))
  else:
    target_res_y = resolution[0] 
    target_res_x = resolution[1] 

  if h == target_res_y and w == target_res_x:
    return copy.deepcopy(data_chunk)

  resize_keys = GTEMPEH_IMAGE_KEYS
  nearest_interpolation = [False] * len(resize_keys)
  nearest_interpolation_keys = ["uv", "depth", "normal", 'sg_parts']
  for k in nearest_interpolation_keys:
    nearest_interpolation[resize_keys.index(k)] = True

  for i in range(len(resize_keys)):
    key = resize_keys[i]
    if key in data_chunk.keys():
      v = data_chunk[key]
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
      resized_data_chunk[key] = v_resized

  # Copy over remaining keys
  for k in set(data_chunk.keys()) - set(resized_data_chunk.keys()):
    resized_data_chunk[k] = data_chunk[k]
  return resized_data_chunk


def unrotate(intrinsics, imgs):
  intrinsics = intrinsics.clone()
  imgs=imgs.clone()
  B, V, C, H, W = imgs.shape
  
  unrotation_mask = intrinsics[:, :, 0, 0] == 0
  Rt = torch.linalg.inv(torch.tensor([ 
            [ 0.,    1,             0           ],
            [-1,    0, H ],
            [ 0,    0,             1           ]
        ]))
  intrinsics[unrotation_mask] = einops.einsum(Rt, intrinsics[unrotation_mask], 'i j, n j k -> n i k')
  assert torch.all(unrotation_mask) or torch.all(~unrotation_mask), 'Inconsistently rotated images not supported'
  if torch.all(unrotation_mask):
    imgs = torch.rot90(imgs, k=-1, dims=(-2, -1))
  return intrinsics, imgs

def public_sample_to_TEMPEH_sample(sample: dict, scale_min=0.9, scale_max=1.1, brightness_sigma=0.33, to_meters=False):  # TODO check val values
  V = len(sample['image'])
  stereo_images = list()
  stereo_camera_intrinsics = list()
  stereo_camera_extrinsics = list()
  stereo_camera_distortions = list()
  stereo_camera_centers = list()
  stereo_images_augmented = list()
  stereo_camera_intrinsics_augmented = list()
  for i in range(V):
    image = einops.rearrange(sample['image'][i], 'c h w -> h w c')
    
    # black bg
    mask = einops.rearrange(sample['fg_mask'][i], 'c h w -> h w c')
    image = image * mask
    
    c2w = geo_util.invert_c2w(sample['Rt'][i:i+1])[0]
    camera = dict(
      intrinsics = sample['K'][i], 
      extrinsics = sample['Rt'][i, :3],
      camera_center = c2w[:3, -1],
      view_direction = c2w[:3, -2],
      image_size=np.array([image.shape[0], image.shape[1]]),  # h, w
      name = str(sample['cameraid'][i]),
      radial_distortion = np.array([0., 0.,], dtype=np.float32)
      )
     
    if camera['image_size'][0] > camera['image_size'][1]:
      # The dataset contains images of landscape and portrait images of resolutions (A x B) and (B x A). 
      # To unify the images for batch handling, rotate all portrait images to landscape.
      image, camera = rotate_image(image, camera)

    # geometric augmentation by random scaling and cropping
    np.random.seed()
    crop_size = (camera['image_size'][0], camera['image_size'][1])
    scale_factor = scale_min + (scale_max - scale_min) * np.random.random()
    h_offset, w_offset = get_random_crop_offsets(crop_size, height=camera['image_size'][0], width=camera['image_size'][1])
    image_augmented, intrinsics_augmented = scale_crop(image, crop_size, h_offset, w_offset, scale_factor, K=camera['intrinsics'])

    # random brightness perturbation
    perturb = 1.0 + brightness_sigma * np.random.randn(1,1,3)
    image_augmented = image_augmented * perturb
    image_augmented = np.clip(image_augmented, 0., 1.)

    # normalize rgb
    image = tempeh_normalize_image(image)
    image_augmented = tempeh_normalize_image(image_augmented)

    image = torch.FloatTensor(torch.from_numpy(image.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
    intrinsics = torch.FloatTensor(torch.from_numpy(camera['intrinsics'].astype(np.float32)))
    extrinsics = torch.FloatTensor(torch.from_numpy(camera['extrinsics'].astype(np.float32)))
    radial_distortion = torch.FloatTensor(torch.from_numpy(camera['radial_distortion'].astype(np.float32)))
    camera_center = torch.FloatTensor(torch.from_numpy(camera['camera_center'].astype(np.float32)))

    image_augmented = torch.FloatTensor(torch.from_numpy(image_augmented.astype(np.float32))).permute(2,0,1).contiguous() # (3,H,W) range (0,1) only rgb
    intrinsics_augmented = torch.FloatTensor(torch.from_numpy(intrinsics_augmented.astype(np.float32)))

    stereo_images.append(image)
    stereo_camera_intrinsics.append(intrinsics)
    stereo_camera_extrinsics.append(extrinsics)
    stereo_camera_distortions.append(radial_distortion)
    stereo_camera_centers.append(camera_center)
    stereo_images_augmented.append(image_augmented)
    stereo_camera_intrinsics_augmented.append(intrinsics_augmented)

  stereo_images = torch.stack(stereo_images, dim=0)
  stereo_camera_intrinsics = torch.stack(stereo_camera_intrinsics, dim=0)
  stereo_camera_extrinsics = torch.stack(stereo_camera_extrinsics, dim=0)
  stereo_camera_distortions = torch.stack(stereo_camera_distortions, dim=0)
  stereo_camera_centers = torch.stack(stereo_camera_centers, dim=0)
  stereo_images_augmented = torch.stack(stereo_images_augmented, dim=0)
  stereo_camera_intrinsics_augmented = torch.stack(stereo_camera_intrinsics_augmented, dim=0)

  data = {
    # img
    'color_images': [],
    'stereo_images': stereo_images,
    'color_images_augmented': [],
    'stereo_images_augmented': stereo_images_augmented,

    # camera
    'color_camera_intrinsics': [],
    'color_camera_extrinsics': [],
    'color_camera_distortions': [],
    'color_camera_centers': [],
    'stereo_camera_intrinsics': stereo_camera_intrinsics,
    'stereo_camera_extrinsics': stereo_camera_extrinsics,
    'stereo_camera_distortions': stereo_camera_distortions,
    'stereo_camera_centers': stereo_camera_centers,
    
    'color_camera_intrinsics_augmented': [],
    'stereo_camera_intrinsics_augmented': stereo_camera_intrinsics_augmented,

    # meta
    # 'index': index,
    'subject': sample['subjectid'],
    'sequence': sample['sequenceid'],
    'frame': sample['frameid'],  
    'camera': sample['cameraid'],

    # meshes
    'v_scan': torch.from_numpy(sample['verts'].astype(np.float32)),
    'v_registration': torch.from_numpy(sample['verts'].astype(np.float32)),
    'f_registration': torch.from_numpy(ava_template_mesh_info['vi'].astype(np.int64)),
    'v_reg_sampled': torch.from_numpy(sample['verts'].astype(np.float32)),
    'f_reg_sampled': torch.from_numpy(ava_template_mesh_info['vi'].astype(np.int64)),
    'v_reg_global': torch.from_numpy(sample['stage1verts'].astype(np.float32)),
    'f_reg_global': torch.from_numpy(ava_template_mesh_info['vi'].astype(np.int64)),
  }

  if not to_meters:
     # from m -> mm
     data['stereo_camera_centers'] *= 1000
     data['stereo_camera_extrinsics'][:, :3, -1] = data['stereo_camera_extrinsics'][:, :3, -1] * 1000
     assert len(data['color_camera_centers']) == 0, 'Conversion not implemented for color cameras'
     data['v_scan'] *= 1000.0  
     data['v_registration'] *= 1000.0
     data['v_reg_sampled'] *= 1000.0 
     data['v_reg_global'] *= 1000.0  

  # Invert z and y axis
  rot_mat = torch.tensor([[1., 0, 0, 0.],
                          [0., -1, 0, 0],
                          [0, 0, -1, 0], 
                          [0., 0., 0. , 1]], dtype=torch.float32)
  assert len(data['color_camera_centers']) == 0, 'Conversion not implemented for color cameras'
  data['stereo_camera_centers'] = (rot_mat[:3, :3] @ data['stereo_camera_centers'].T).T
  data['stereo_camera_extrinsics'] = data['stereo_camera_extrinsics'] @ rot_mat.T[None]
  data['v_scan'] = (rot_mat[:3, :3] @ data['v_scan'].T).T
  data['v_registration'] = (rot_mat[:3, :3] @ data['v_registration'].T).T
  data['v_reg_sampled'] = (rot_mat[:3, :3] @ data['v_reg_sampled'].T).T
  data['v_reg_global'] = (rot_mat[:3, :3] @ data['v_reg_global'].T).T

  return data

def public_sample_to_gtempeh_sample(sample:dict, geometry_scale_factor=1.):
  """ Converts samples from Ava and Nersemble to gtempeh samples

  Args:
    sample: dataset sample with keys
      verts
      stage1verts
      Rt
      K
      idindex
      camindex
      image
      uv
      bg
      fg_mask
      sg_parts
      frameid
      cameraid
      subjectid
      sequenceid
      scene_rotation
      scene_center
      scene_scale
  
  Returns: gtempeh sample with keys
    'verts', 'stage1verts', 'C2W', 'fxfycxcy', 'image', 'uv', 'bg', 'sg_parts', 'mask', 'frame', 'camera', 'subject', 'sequence', 'scene_rotation', 'scene_center', 'scene_scale'


  """
  gtempeh_sample = dict()
  H, W = sample['image'].shape[-2:]
  for k, v in sample.items():
    if isinstance(v, np.ndarray):
      v = torch.from_numpy(v)

    if k == 'fg_mask':
      k = 'mask'
    
    if k == 'K': 
      assert torch.all(v[:, 0, 1] == 0) and torch.all(v[:, 1, 0] == 0), 'Expects skew-free camera'
      fx = v[:, 0, 0] / W
      fy = v[:, 1, 1] / H
      cx = v[:, 0, 2] / W
      cy = v[:, 1, 2] / H
      k = 'fxfycxcy'
      v = torch.stack([fx,fy,cx,cy], dim=-1)
    
    if k == 'Rt': 
      C2W = geo_util.invert_c2w(v)
      C2W[..., :3, -1] = C2W[..., :3, -1] * geometry_scale_factor
      k = 'C2W'
      v = C2W
  
    if k in ['verts', 'stage1verts']: 
      v = v * geometry_scale_factor
    
    if k == 'subjectid':
      k = 'subject'
    
    if k == 'frameid':
      k = 'frame'

    if k == 'cameraid':
      k = 'camera'
    
    if k == 'sequenceid':
      k = 'sequence'

    gtempeh_sample[k] = v
  return gtempeh_sample


def str_to_color(s: str) -> List[float]:
  """Converts a string to a list of 3 floats 0...1 representing a color."""
  if s == "white":
    return [1.0, 1.0, 1.0]
  elif s == "black":
    return [0.0, 0.0, 0.0]
  elif s == "random":
    return torch.rand(3).tolist()
  else:
    raise ValueError(f"Unknown color: {s}")


def colorize_bg(data: dict, color: List[float]) -> dict:
  """Colorizes the background (applies color both to bg and image).

  Args:
    data: A dictionary of data with keys:
      image: ((B), V, 3, H, W) 0...1
      bg: ((B), V, 3, H, W) 0...1
      mask: ((B),V, 1, H, W) 0...1
  """

  colorized_data = dict()
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


def resize_sample_tf(
    data: Dict[str, tf.Tensor],
    resize_keys: List[str],
    nearest_interpolation_keys: Optional[List[str]] = None,
    resolution: Optional[int] = None,
) -> Dict[str, tf.Tensor]:
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
    data: Dict[str, tf.Tensor],
    nviews: Optional[int] = None,
    ninputviews: Optional[int] = None,
    even_sampling: bool = False,
) -> Dict[str, tf.Tensor]:
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

def apply_rotation_scale_center_to_points(points: np.ndarray, rotation:np.ndarray, scale: np.ndarray, center:np.ndarray):
        '''
        
        Args:
            points: ([B,] N,3)
            rotation: ([B,] 3, 3)
            scale: ([B,])
            center: ([B,] 3)
        
        Returns:
            rotated, scaled and centered points (N, 3)
        '''
        has_batch_dim = len(points.shape) == 3
        if not has_batch_dim:
           points = points[None]
           rotation = rotation[None]
           scale = scale[None]
           center = center[None]

        points = einops.einsum(rotation, points, 'b i j, b n j -> b n i')*einops.rearrange(scale, 'b -> b 1 1') - einops.rearrange(center, 'b c -> b 1 c')

        if not has_batch_dim:
           points = points[0]
        
        return points

def apply_inv_rotation_scale_center_to_points(points: np.ndarray, rotation:np.ndarray, scale: np.ndarray, center:np.ndarray):
        '''
        
        Args:
            points: ([B,] N,3)
            rotation: ([B,] 3, 3)
            scale: ([B,])
            center: ([B,] 3)
        
        Returns:
            rotated, scaled and centered points (N, 3)
        '''
        has_batch_dim = len(points.shape) == 3
        if not has_batch_dim:
           points = points[None]
           rotation = rotation[None]
           scale = scale[None]
           center = center[None]
        
        points = points + einops.rearrange(center, 'b c -> b 1 c')
        points = einops.einsum(rotation.swapaxes(-1,-2), points , 'b i j, b n j -> b n i')
        points = points / einops.rearrange(scale, 'b -> b 1 1')

        if not has_batch_dim:
           points = points[0]
        
        return points


def random_view_selection_torch(
    data: Dict[str, torch.Tensor],
    nviews: Optional[int] = None,
    ninputviews: Optional[int] = None,
    even_sampling: bool = False,
) -> Dict[str, torch.Tensor]:
  """Randomly selects a subset of the views from the multi-view data.

  Selects a subset of views from the augmented images and augmented camera
  parameters. The original images and camera intrinsics in the data dict remain
  unchanged.

  if even_sampling:
    input views (first ninputviews) are sampled evenly (deterministic)from the
    beginning of the
    view sequence and ensures that the target views lie between them
  else:
    input views are sampled randomly

  Args:
    data: Multi-view images and camera data. Can be a single sample or a batch.
    nviews: Number of views to select.
    ninputviews: Number of input views to select.
    deterministic: Whether to use a deterministic view selection.

  Returns:
    A dict with a subset of views for the augmented images and cameras.
  """
  is_batch = len(data["image"].shape) > 4
  V = data["image"].shape[-4]
  if nviews is None:
    nviews = V
  if ninputviews is None:
    ninputviews = nviews

  if even_sampling:
    view_idcs = torch.round(torch.linspace(0, V - 1, nviews)).long()
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
  else:
    view_samples = torch.randperm(V)[:nviews]

  out_dict = dict()
  for k, v in data.items():
    if k not in ["subject", "sequence", "frame", "verts", "stage1verts", 'idindex', 'scene_rotation', 'scene_center', 'scene_scale', 'vertex_masks']:
      v = torch.index_select(v, dim=1 if is_batch else 0, index=view_samples)
    out_dict[k] = v
  return out_dict


def visualize_camera_grid(samples: List[dict], sample_names:List, outpath: str):
    """
    Visualizes a sample's camera grid:
    - 2D grid of images with camera IDs (saved as PNG)
    - 3D visualization of camera extrinsics and mesh (saved as HTML)

    Args:
        sample: Dictionary with keys:
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

    
    for sample, sample_name in zip(samples, sample_names):
      images = sample["image"]      # (V, 3, H, W)
      C2W = sample["C2W"]           # (V, 4, 4)
      verts = sample["verts"]       # (N, 3)
      camera_ids = list(map(str, sample["camera"].tolist())) # list of length V

      # Ensure output directory exists
      file_helper.Path(outpath).parent.mkdir(parents=True, exist_ok=True)

      V = images.shape[0]
      grid_size = math.ceil(math.sqrt(V))

      # --- 2D image grid ---
      fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*3, grid_size*3))
      axes = axes.flatten()

      for i in range(grid_size * grid_size):
          ax = axes[i]
          ax.axis('off')
          if i < V:
              img = images[i].permute(1, 2, 0).cpu().numpy()
              ax.imshow(img)
              ax.set_title(str(camera_ids[i]))

      plt.tight_layout()
      plt.savefig(f"{outpath}_grid_{sample_name}.png")
      plt.close(fig)
      print(f'Saved 2D grid to {outpath}_grid_{sample_name}.png')

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
          c2w = C2W[i].cpu().numpy()
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
                  name=f'Cam{camera_ids[i]}_{sample_name}_{name}'
              ))

          # Add floating text with camera id
          fig_3d.add_trace(go.Scatter3d(
              x=[origin[0]], y=[origin[1]], z=[origin[2]],
              mode='text',
              text=[f"{camera_ids[i]}_{sample_name}_az{int(az)}_el{int(el)}"],
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



def sample_cameras_from_angles(camera_positions: np.ndarray, sample_angles: list, center=np.array([0, 0, 0.]), unique=True, ncams_per_angle=1):
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
                    indices.append(idx)
                    used[idx] = True
                    counter += 1
                    if counter == ncams_per_angle:
                      break
        else:
            indices.extend(sorted_idx[:ncams_per_angle].tolist())

    return indices


def path_2_zippath(path:str):
   zipfile_path, content_path = path.split('.zip/')
   return ZipPath(zipfile_path + '.zip', content_path)

def visualize_sample(
    sample: dict, outpath: str, faces: Optional[np.ndarray] = None, s=6, show=False
):
  """Visualizes a sample.

  Args:
    sample: A dictionary of data.
    faces: A numpy array of faces. (F, 3)
    output_dir: The output directory.
    s: The size of the figure.
    show: Whether to show the figure.
  """
  if show:
    plt.switch_backend("TkAgg")
  # visualize sample
  image = sample["image"]  # (V, 3, H, W), 0-1
  mask = sample["mask"]  # (V, 1, H, W), 0-1
  bg = sample["bg"]  # (V, 3, H, W), 0-1
  C2W = sample["C2W"]  # (V, 4, 4),
  W2C = torch.linalg.inv(C2W)  # (V, 4, 4)
  # normal = sample["normal"]  # (V, 3, H, W), 0-1
  uv = sample["uv"]  # (V, 3, H, W), 0-1
  gt_verts = sample["verts"]  # (N, 3)
  stage1_verts = sample["stage1verts"]  # (N, 3)
  fxfycxcy = sample["fxfycxcy"]  # (V, 4)
  H, W = image.shape[-2:]

  gt_verts_screen = geo_util.project_points_to_screen(gt_verts[None], c2w=C2W[None], fxfycxcy=fxfycxcy[None], H=H, W=W)[0]
  stage1_verts_screen = geo_util.project_points_to_screen(stage1_verts[None], c2w=C2W[None], fxfycxcy=fxfycxcy[None], H=H, W=W)[0]

  V, _, H, W = image.shape
  fig, axes = plt.subplots(
      7, V, figsize=(V * s, 7 * s), squeeze=False
  )
  for v in range(V):
    axes[0, v].imshow(image[v].numpy().transpose(1, 2, 0))
    axes[0, v].set_title(f"Image {v}")
    axes[0, v].axis("off")
    axes[1, v].imshow(bg[v].numpy().transpose(1, 2, 0))
    axes[1, v].set_title(f"Background {v}")
    axes[1, v].axis("off")
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
        stage1_verts_screen[v, :, 0]
        - 0.5,  # correcting for matplotlib assumes top left pixel center to have coords 0.0, 0.0 but visual_sfm assumes 0.5, 0.5
        stage1_verts_screen[v, :, 1] - 0.5,
        color="red",
        s=0.5,
    )
    axes[3, v].set_title(f"stage1_mesh_overlay {v}")
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
    axes[6, v].imshow(sample['uv'][v].numpy().transpose(1, 2, 0), alpha=0.7)
    axes[6, v].set_title(f"UV Map {v}")
    axes[6, v].axis("off")
  plt.tight_layout()
  if show:
    plt.show()

  output_filename = outpath
  file_helper.Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
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
                x=stage1_verts[:, 0],
                y=stage1_verts[:, 1],
                z=stage1_verts[:, 2],
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


CELL_2_CNS_CELL = {
    "gc": "gc-d",
    "gd": "gc-d",
    "gg": "gc-d",
    "la": "li-d",
    "lb": "li-d",
    "lj": "li-d",
    "lm": "li-d",
    "lq": "li-d",
    "td": "tp-d",
    "yurnoaa": "ro-d",
}


def resolve_cns_path(path):
  """Resolves a CNS path to a specific cell.

  replaces magic string {CNS_CELL} with the current compute cell's CNS cell
  works with both gpath and str paths
  """

  is_path = isinstance(path, file_helper.Path)
  if is_path:
    path = str(path)

  current_compute_cell = os.environ.get("BORG_CELL", "")
  storage_cell = CELL_2_CNS_CELL.get(current_compute_cell, "")
  path = path.replace("{CNS_CELL}", storage_cell)
  if is_path:
    path = file_helper.Path(path)

  return path
