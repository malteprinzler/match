from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional
from typing import List
import gin


@gin.configurable
@dataclass
class Options:
  base_folder: str = ""
  resume_from_iter: int | None = -1
  seed: int = 0
  max_train_steps: int = 1_000_000
  geom_only_steps: int = 0
  n_vissamples: int = 5
  max_val_steps: int = 100
  debug_few_samples: int = -1
  find_unused_parameters: bool = False
  pin_memory: bool = True
  use_ema: bool = False
  scale_lr: bool = False
  max_grad_norm: float = 1.0
  gradient_accumulation_steps: int = 1
  mixed_precision: str = "fp16"
  allow_tf32: bool = True
  load_pretrained_model: str | None = None
  load_pretrained_model_ckpt: int = -1
  load_pretrained_model_strict: bool = True
  initial_eval: bool = False
  log_level: str = "INFO"
  train_total_batch_size: int | None = None  # will be calculated later
  log_freq: int = 100
  early_eval: int = 1000
  early_eval_freq: int = 1000
  eval_freq: int = 5000
  save_freq: int = 5000
  vertex_group_only: str | None = None  # 'hockey_mask' for face region only
  use_feature_net: bool = False
  image_feature_net_type: str = ""
  image_feature_net_kwargs: dict = field(default_factory=lambda: {})

  noloss_masks: List[str] = field(default_factory=lambda: [])
  noloss_mask_dilation: int = 0
  noloss_mask_threshold: float = 0.1
  noloss_mask_strength: float = 1.0

  val_input_idcs: Optional[List[int]] = field(default_factory=lambda: None)

  # GaussianSplat Renderer
  two_dgs: bool = True

  optimizer: dict = field(
      default_factory=lambda: {
          "name": "adamw",
          "lr": 4e-5,  # @mprinzler changed from 0.0004,
          "betas": [0.9, 0.95],
          "weight_decay": 0.05,
      }
  )

  lr_scheduler: dict = field(
      default_factory=lambda: {
          "name": "cosine_warmup",
          "num_warmup_steps": 1000,
          "total_steps": 1_000_000,
      }
  )

  # Dataset
  overfit: bool = False
  overfit_n_batches: int = 1
  rot90: bool = False
  train_loader_params: dict = field(default_factory=lambda:{}) # entry must have keys: 'cls' and 'kwargs' 
  val_loader_params: list = field(default_factory=lambda:[]) # list of entries each of which must have keys: 'cls', 'kwargs', and 'name'
  seq_loader_params: list = field(default_factory=lambda:[]) # list of entries each of which must have keys: 'cls', 'kwargs', and 'name'
  segmentation_labels_path: str = ''

  input_res: int = 128  # @mprinzler changed from 256 to 128
  input_bg: str = "black"  # '' means original background
  output_bg: str = "black"  # '' means white bg

  ## Camera
  num_input_views: int = 4
  num_views: int = 8  # controls how many views dataset loads
  num_render_views: int | None = (
      None  # controls how many views are actually rendered for training; if none, renders all views
  )
 
  # MATCH model
  template_mesh_path: str = 'assets/ava256/face_topology_cleaned.obj'
  template_mesh_vertex_groups_path: str = 'assets/ava256/vertex_groups.json'
  assets_path: str = 'assets'
  uv_res: int = 256
  use_sapiens_features: bool = False
  sapiens_ckpt_path: str = 'assets/sapiens/checkpoints/torchscript/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2'
  sapiens_prediction_tiles_sqrt: int = 2
  n_neighbors: int = 10
  init_gs_std: float = 0.07

  ## Transformer
  llama_style: bool = True
  uv_patch_size: int = 8
  img_patch_size: int = 8
  dim: int = 512
  num_blocks: int = 12
  num_heads: int = 8
  grad_checkpoint: bool = True

  ## Rendering
  render_type: Literal[
      "default",
      "deferred",
  ] = "default"
  deferred_bp_patch_size: int = 64
  znear: float = 0.01
  zfar: float = 100.0
  scale_min: float = 0.0005
  scale_max: float = 0.02
  skip_connection: bool = False  # predict residuals to input gaussian params

  # Training
  restart_after_eval_frequency: int | None = None
  nccl_timeout: float = 20.0

  # Loss weights
  output_res: int | None = None
  render_weight: List[float] | float = 1.0
  lpips_weight: List[float] | float = 1.0
  lpips_warmup_start: int = 0
  lpips_warmup_end: int = 0
  xyz_weight: List[float] | float = 1e-3
  mesh_vert_weight: List[float] | float = 1e-3
  scale_weight: List[float] | float = 1e-3
  opacity_weight: List[float] | float = 1e-3
  scale_target: List[float] | float = 0.001
  opacity_target: List[float] | float = 0.7
  ssim_weight: List[float] | float = 0.
  l1_weight: List[float] | float = 0.
  geometry_loss_weights_by_part:Dict[str, float] = field(default_factory=lambda:dict(__default__=1.))

  # Visualization
  vis_idx: int = 0

def _update_opt(opt: Options, **kwargs) -> Options:
  new_opt = deepcopy(opt)
  for k, v in kwargs.items():
    setattr(new_opt, k, v)
  return new_opt
