from match.models.sapiens_encoder import SapiensEncoder
import json
from typing import Dict, Tuple
import einops
import lpips
import numpy as np
import pudb
from match.models import image_feature_net
from match.models.gsplat_renderer import gsplat_renderer
from match.models.networks.attention import *
from match.options import Options
from match.utils import file_util, image_util, mesh_util, op_util, data_util, general_util, geo_util, uv_util
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as tF
import trimesh
import torchmetrics
from functools import partial


class MatchModel(nn.Module):

  def __init__(self, opt: Options):
    super().__init__()
    self.opt = opt

    self.uv_res = opt.uv_res
    self.uv_inter_res = self.uv_res // opt.uv_patch_size
    self.img_patch_size = opt.img_patch_size
    self.uv_patch_size = opt.uv_patch_size
    self.skip_connection = opt.skip_connection
    self.use_sapiens_features = opt.use_sapiens_features

    # Image tokenizer
    in_channels = 3  # RGB

    # Image Encoder
    if opt.use_feature_net:
      feature_net_container = image_feature_net.ImageFeatureNetContainer(
          image_feature_net.ImageFeatureNetType[opt.image_feature_net_type]
      )
      self.feature_net = feature_net_container.model(
          in_channels=in_channels,
          image_downscale_factor=self.img_patch_size,
          **opt.image_feature_net_kwargs,
      )
      x_embedder_in_channels = (
          6 * self.img_patch_size**2 + self.feature_net._out_channels
      )  # +6 Plucker
    else:
      self.feature_net = None
      x_embedder_in_channels = (in_channels + 6) * (
          opt.img_patch_size**2
      )  # +6 Plucker
    self.x_embedder = nn.Linear(x_embedder_in_channels, opt.dim)

    self.sapiens_embedder = None
    self.sapiens_encoder = None
    if self.use_sapiens_features:
      self.sapiens_encoder = SapiensEncoder(ckpt=opt.sapiens_ckpt_path, device=torch.device('cpu'), patch_size=self.img_patch_size, dtype=torch.bfloat16, 
                                           prediction_tiles_sqrt=opt.sapiens_prediction_tiles_sqrt)
      self.sapiens_embedder = nn.Linear(opt.dim + self.sapiens_encoder.C, opt.dim)  # merges the sapiens features into the tokens

    # UV Encoder / Tokens
    uv_patch_embedder_in_channels = (
        3 + 3  # 3 positions + 3 colors
    ) * opt.uv_patch_size**2  # 2 channels for uv, 3 channels for xyz
    self.uv_patch_embedder = nn.Linear(uv_patch_embedder_in_channels+opt.dim, opt.dim)

    # assert opt.dim - uv_patch_embedder_in_channels>0
    self.uv_tokens = nn.Parameter(
        torch.randn(
            self.uv_inter_res**2, opt.dim, dtype=self.x_embedder.weight.dtype
        )
    )

    # Transformer backbone
    self.transformer = CorrespondenceAwareTransformer(
        opt.num_blocks,
        opt.dim,
        opt.num_heads,
        llama_style=opt.llama_style,
    )
    if opt.grad_checkpoint:
      self.transformer.set_grad_checkpointing()

    # Output heads
    self.out_xyz = nn.Linear(opt.dim, 3 * (opt.uv_patch_size**2), bias=False)
    self.out_rgb = nn.Linear(opt.dim, 3 * (opt.uv_patch_size**2), bias=False)
    self.out_scale = nn.Linear(opt.dim, 3 * (opt.uv_patch_size**2), bias=False)
    self.out_rotation = nn.Linear(
        opt.dim, 4 * (opt.uv_patch_size**2), bias=False
    )
    self.out_opacity = nn.Linear(
        opt.dim, 1 * (opt.uv_patch_size**2), bias=False
    )
    self.ln_out = nn.LayerNorm(opt.dim)

    self._register_uvcaches(
        self.uv_res,
        opt.vertex_group_only,
    )

    # Rendering
    self.gs_renderer = gsplat_renderer.GSplatRenderer(
        twoDgs=opt.two_dgs
    )

    # Initialize weights
    nn.init.xavier_uniform_(self.x_embedder.weight)
    nn.init.zeros_(self.x_embedder.bias)
    # nn.init.zeros_(self.out_depth.weight)  # zero init.
    nn.init.xavier_uniform_(self.out_xyz.weight)  # zero init.
    self.out_xyz.weight.data = self.out_xyz.weight.data * opt.init_gs_std
    nn.init.xavier_uniform_(self.out_rgb.weight)
    nn.init.zeros_(self.out_scale.weight)  # zero init.
    nn.init.xavier_uniform_(self.out_rotation.weight)
    nn.init.zeros_(self.out_opacity.weight)  # zero init.
    # if self.uv_patch_embedder is not None:
    #   nn.init.xavier_uniform_(self.uv_patch_embedder.weight)
    #   nn.init.zeros_(self.uv_patch_embedder.bias)
    if self.use_sapiens_features:
      nn.init.xavier_uniform_(self.sapiens_embedder.weight)
      nn.init.zeros_(self.sapiens_embedder.bias)

    with file_util.open_file(opt.segmentation_labels_path, 'r') as f:
      self.segmentation_labels = json.load(f)

    self.ssim = partial(torchmetrics.functional.image.structural_similarity_index_measure, data_range=(0., 1.), reduction='none')

    ava2flame_mapping = dict(np.load(f'{opt.assets_path}/flame/mapping_ava2flame.npz'))
    for k, v in ava2flame_mapping.items():
        ava2flame_mapping[k] = torch.from_numpy(v)
    self.ava2flame_mapping = ava2flame_mapping
      
    
  def to(self, *args, **kwargs):
    device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
    super().to(*args, **kwargs)
    if self.sapiens_encoder is not None:
      self.sapiens_encoder.to(device)
    return self
    

  def get_uv_rgb(self, data:dict, uv_xyz, correspondence_idcs, correspondence_scores, n_views = 5):
    """
    for each texel in uv space, calculates color reprojections from input images

    Args:

    Returns:
      reprojected color texture (B, 3, H_uv, W_uv)
    """
    # for each patch get most relevant views 
    B, V, C, H, W = data['image'].shape
    n_views = min(V, n_views)
    device = uv_xyz.device

    corr_vidx = uv_util.unflatten_patch_idcs(correspondence_idcs, H=H, W=W, patch_size=self.opt.img_patch_size)[0]

    uv_patch_viewscores = torch.zeros((B, self.uv_inter_res, self.uv_inter_res, V), device=device, dtype=correspondence_scores.dtype)
    uv_patch_viewscores.scatter_reduce_(dim=-1, index=corr_vidx, src=correspondence_scores, reduce='sum')  # B, uv_inter_res, uv_inter_res, V

    uv_patch_repr_viewscores, uv_patch_repr_viewidcs = torch.sort(uv_patch_viewscores, dim=-1, descending=True)
    uv_patch_repr_viewscores = uv_patch_repr_viewscores[..., :n_views]  # B, uv_inter_res, uv_inter_res, n_views_repr
    uv_patch_repr_viewidcs = uv_patch_repr_viewidcs[..., :n_views]  # B, uv_inter_res, uv_inter_res, n_views_repr

    # project 3d locations to relevant view coords
    # B, n_views_repr, c, h, w
    b_helper = einops.repeat(torch.arange(B, device=device), 
                              'b -> b n_patch_v n_patch_h n_views', 
                              n_patch_v = self.uv_inter_res, 
                              n_patch_h = self.uv_inter_res, 
                              n_views = n_views)  # B, uv_inter_res, uv_inter_res, n_views_repr
    uv_patch_repr_c2w = data['C2W'][b_helper, uv_patch_repr_viewidcs]  # B, uv_inter_res, uv_inter_res, n_views_repr, 4, 4
    uv_patch_repr_fxfycxcy =  data['fxfycxcy'][b_helper, uv_patch_repr_viewidcs]  # B, uv_inter_res, uv_inter_res, n_views_repr, 4

    uv_patch_repr_c2w_flat = einops.rearrange(uv_patch_repr_c2w, 'B n_patch_v n_patch_h n_views_repr c1 c2 -> (B n_patch_v n_patch_h) n_views_repr c1 c2')
    uv_patch_repr_fxfycxcy_flat = einops.rearrange(uv_patch_repr_fxfycxcy, 'B n_patch_v n_patch_h n_views_repr c1 -> (B n_patch_v n_patch_h) n_views_repr c1')
    uv_patch_verts = einops.rearrange(uv_xyz, 'b c (n_patch_v patch_h) (n_patch_h patch_w) -> (b n_patch_v n_patch_h) (patch_h patch_w) c', 
                                      n_patch_v = self.uv_inter_res, 
                                      n_patch_h=self.uv_inter_res, 
                                      patch_h=self.uv_patch_size, 
                                      patch_w=self.uv_patch_size, b=B)
    uv_patch_repr_screen_coords = geo_util.project_points_to_screen(
        points=uv_patch_verts, 
        c2w=uv_patch_repr_c2w_flat, 
        fxfycxcy=uv_patch_repr_fxfycxcy_flat, 
        H=H, W=W)    # (B n_patch_v n_patch_h) n_views_repr nverts_per_patch 3
    uv_patch_repr_query_coords = torch.round(uv_patch_repr_screen_coords[..., :2] - .5).to(torch.int)  # stores indices at which to sample the images (effectively nearest neighbor sampling). -.5 because we assume top left corner of top left pixel to have coordinate 0 but for tensor indexing this should be -.5

    # querying the colors
    uv_patch_repr_query_coords = einops.rearrange(
      uv_patch_repr_query_coords, '(B n_patch_v n_patch_h) n_views_repr nverts_per_patch c-> B n_patch_v n_patch_h n_views_repr nverts_per_patch c', 
      B=B, 
      n_patch_v=self.uv_inter_res, 
      n_patch_h=self.uv_inter_res)
    b_helper = einops.repeat(torch.arange(B, device=device), 'b -> b n_patch_v n_patch_h n_views verts_per_patch', 
      n_patch_v = self.uv_inter_res, 
      n_patch_h = self.uv_inter_res, 
      n_views = n_views, 
      verts_per_patch = self.uv_patch_size**2)  # B, uv_inter_res, uv_inter_res, n_views_repr, nverts_per_patch
    v_helper = einops.repeat(uv_patch_repr_viewidcs, 'B n_patch_v n_patch_h n_views_repr -> B n_patch_v n_patch_h n_views_repr verts_per_patch', verts_per_patch = self.uv_patch_size**2)  # B, uv_inter_res, uv_inter_res, n_views_repr, nverts_per_patch
    uv_patch_repr_colors = data['image'][b_helper, v_helper, :, uv_patch_repr_query_coords[..., 1].clamp(0, H-1), uv_patch_repr_query_coords[..., 0].clamp(0, W-1)]  # B n_patch_v n_patch_h nviews_repr nverts_per_patch 3

    # masking out invalid colors (projected outside of image or 0 correspondence score)
    invalid_mask = torch.any(uv_patch_repr_query_coords<0, dim=-1) \
      | (uv_patch_repr_query_coords[..., 0]>(W-1)) \
      | (uv_patch_repr_query_coords[..., 1]>(H-1)) \
      | (uv_patch_repr_viewscores[..., None] == 0)  # B n_patch_v n_patch_h nviews_repr nverts_per_patch 
    uv_patch_repr_colors[invalid_mask] = float('nan')

    # aggregating reprojected colors through median, fill nans with 0
    uv_patch_repr_colors = torch.nanmedian(uv_patch_repr_colors, dim=-3).values  # B n_patch_v n_patch_h nverts_per_patch 3
    uv_patch_repr_colors = torch.nan_to_num(uv_patch_repr_colors)

    # reshaping back to uv texture
    uv_patch_repr_colors = einops.rearrange(uv_patch_repr_colors, 'B n_patch_v n_patch_h (patch_height patch_width) c -> B c (n_patch_v patch_height) (n_patch_h patch_width)', patch_height = self.uv_patch_size, patch_width=self.uv_patch_size)

    # # visualizing reprojected colors
    # import matplotlib.pyplot as plt
    # fig = plt.figure(figsize=(6,6))
    # plt.imshow(uv_patch_repr_colors[0].permute(1,2,0).cpu().numpy())
    # plt.savefig('demos/color_reprojection.jpg')
    # plt.close(fig)

    return uv_patch_repr_colors  # B 3 H_uv W_uv

  def _get_uv_tokens(
      self, data: Dict[str, Tensor], correspondence_idcs: torch.Tensor, correspondence_scores: torch.Tensor, dtype: torch.dtype = torch.float32, return_uv_xyz_rgb=False
  ):
    """Calculates uv tokens from batch.

    If self.uv_tokens is not None, then explicit self.uv_tokens are used,
    otherwise uv tokens are calculated from using the uv coordinates and
    stage1verts.

    Args:
      data: A dictionary of data with keys: - vert: (B, Nverts, 3)
      dtype: The data type to use.

    Returns:
      x: uv patch tokens of shape (B, N, D)
    """
    b = len(data["verts"])

    # tokens from verts
    verts = data["coarse_verts"].to(dtype)  # (B, Nverts, 3)
    try:
      uv_xyz = self._uv_xyz_from_verts(verts)
    except Exception as e:
      print(f'FAILED uv_xyz creation for sample {data["idindex"]}, verts.shape: {verts.shape}')
      raise e
    uv_rgb = self.get_uv_rgb(data=data, correspondence_idcs=correspondence_idcs, correspondence_scores=correspondence_scores, uv_xyz=uv_xyz)
    x = torch.cat((uv_xyz, uv_rgb), dim=-3)
    x = op_util.patchify(x, self.opt.uv_patch_size)  # (B, N, C)
    learnable_pe = einops.repeat(self.uv_tokens, 'n c -> b n c', b=b)
    x = torch.cat((x, learnable_pe), dim=-1)
    x = self.uv_patch_embedder(x)

    if return_uv_xyz_rgb:
      return x, uv_xyz, uv_rgb
    else:
      return x

  def _get_img_tokens(
      self, data: Dict[str, Tensor], dtype: torch.dtype = torch.float32
  ):
    """Args:

      data: A dictionary of data with keys:
        - image: (B, V, 3, H, W)  normalized to [0...1]
        - C2W: (B, V, 4, 4)
        - fxfycxcy: (B, V, 4)
        - (sapiens_features) : (B, V, C, H', W')
      dtype: The data type to use.
    Returns:
      x: image patch tokens of shape (B, N, D)
    """
    color_name = "image"

    images = data[color_name].to(dtype)  # (B, V, 3, H, W)
    C2W = data["C2W"].to(dtype)  # (B, V, 4, 4)
    fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, V, 4)

    # Input views
    B, V, _, H, W = images.shape
    input_images = images
    input_C2W = C2W
    input_fxfycxcy = fxfycxcy

    input_images = input_images * 2.0 - 1.0
    input_images = image_util.pad_image_to_fit_patchification(
        input_images, self.img_patch_size
    )

    # sapiens feature extraction
    sapiens_features = None
    if self.use_sapiens_features:
      sapiens_features = self.sapiens_encoder(einops.rearrange(input_images*.5 + .5, 'b v c h w -> (b v) c h w')) 
      sapiens_features = einops.rearrange(sapiens_features, '(b v) c h w -> b v c h w', v=V)

    if self.feature_net is None:
      input_images = einops.rearrange(
          input_images,
          "b v c (n_patch_vert hpatch) (n_patch_horiz wpatch) -> b v (hpatch"
          " wpatch c) n_patch_vert n_patch_horiz",
          hpatch=self.img_patch_size,
          wpatch=self.img_patch_size,
      )  # (B, V, C, N_patch_vert, N_patch_horiz)

    else:
      b, v = input_images.shape[:2]
      input_images = einops.rearrange(
          self.feature_net(
              einops.rearrange(input_images, "b v c h w -> (b v) h w c")
          ),
          "(b v) h w c -> b v c h w",
          b=b,
          v=v,
      )  # (B, V, C, N_patch_vert, N_patch_horiz)
    n_patches_v, n_patches_h = input_images.shape[-2:]

    # Plucker
    plucker, _ = geo_util.plucker_ray(
        H, W, input_C2W, input_fxfycxcy
    )  # (B, V_in, 6, H, W)
    plucker = image_util.pad_image_to_fit_patchification(
        plucker, self.img_patch_size
    )
    plucker = einops.rearrange(
        plucker,
        "b v c (n_patch_vert hpatch) (n_patch_horiz wpatch) -> b v (hpatch"
        " wpatch c) n_patch_vert n_patch_horiz",
        hpatch=self.img_patch_size,
        wpatch=self.img_patch_size,
        n_patch_vert=n_patches_v,
        n_patch_horiz=n_patches_h,
    )

    # Encoding into tokens
    images_plucker = torch.cat([input_images, plucker], dim=2)
    images_plucker = einops.rearrange(
        images_plucker, "b v c h w -> b v (h w) c"
    )  # (B, V_in, N, D)
    x = self.x_embedder(images_plucker)  # (B, V_in, N, D)

    # incorporating sapiens
    if self.use_sapiens_features:
      sapiens_features = einops.rearrange(sapiens_features.to(x), 'b v c n_patches_v n_patches_h -> b v (n_patches_v n_patches_h) c', n_patches_v=n_patches_v, n_patches_h=n_patches_h)
      x = torch.cat((x, sapiens_features), dim=-1)
      x = self.sapiens_embedder(x)


    x = einops.rearrange(x, "b v n d -> b (v n) d")
    return x

  def scale_activation(self, x: Tensor):
    return self.opt.scale_min * x + self.opt.scale_max * (
        1.0 - x
    )  # [0, 1] -> [s_min, s_max]

  def gaussians2mesh(self, gaussian_parameters: dict[str, Tensor]):
    """Converts gaussian parameters to a mesh.

    Args:
      gaussian_parameters: A dictionary of gaussian parameters with keys: - xyz:
        (B, 1, 3, H, W) - ...

    Returns:
      verts: A torch tensor of shape (B, Verts, 3)
    """
    xyz = gaussian_parameters["xyz"]
    b = len(xyz)

    xyz = einops.rearrange(xyz, "b 1 c h w -> b c h w")
    triangle_uvs = (
        self.template_triangle_uvs.unsqueeze(0)
        .expand(b, -1, -1, -1)
        .to(xyz)
    )  # (B, N_faces, 3, 2)
    triangle_uvs = triangle_uvs * 2 - 1  # [0, 1] -> [-1, 1]
    triangle_uvs[..., 1] *= -1  # flip y axis: origin at bottom left -> top left
    xyz_triangles = torch.nn.functional.grid_sample(
        xyz,
        triangle_uvs,
        mode="bilinear",
        align_corners=True,  # (-1, -1) refers to center of top-left pixel
        padding_mode="border",
    )  # (B, 3(xyz), N_faces, 3(verts per face))

    # aggregating mesh vertex 3D coordinates for occurences in different triangles
    xyz_triangles = einops.rearrange(xyz_triangles, "b c f v -> b f v c")
    xyz_vert = self.triangle_values_to_vert_values(xyz_triangles)
    return xyz_vert

  def gaussians2flamemesh(self, gaussian_parameters):
    verts_ava = self.gaussians2mesh(gaussian_parameters)
    
    tri_vids = self.ava2flame_mapping['vertex_indices'].to(verts_ava.device)
    bary = self.ava2flame_mapping['barycentric_coordinates'].to(dtype=verts_ava.dtype, device=verts_ava.device)
    flame_verts = einops.einsum(bary, einops.rearrange(verts_ava, 'b n c -> n b c')[tri_vids], 'v i, v i b c -> b v c')
    return flame_verts

  def triangle_values_to_vert_values(
      self, triangle_values, triangle_vert_ids=None, v=None
  ):
    """Aggregates triangle values to vertex values (averaging over duplicate vertex occurences).

    Args:
      triangle_values: A torch tensor of shape (B, N_faces, 3(verts per face),
        C)
      triangle_vert_ids: A torch tensor of shape (N_faces, 3(verts per face))
        with vertex ids for each triangle vertex
      v: Number of vertices in the mesh. If None, it is set to the number of
        vertices in the G-Nome mesh.

    Returns:
      vert_values: A torch tensor of shape (B, V, C)
    """
    b = len(triangle_values)
    v = self.num_verts if v is None else v
    c = triangle_values.shape[-1]

    triangle_values = einops.rearrange(triangle_values, "b f v c -> b (f v) c")
    if triangle_vert_ids is None:
      triangle_vert_ids = self.template_triangles
    triangle_vert_ids = triangle_vert_ids.to(torch.int64)
    triangle_vert_ids = einops.rearrange(triangle_vert_ids, "f v -> 1 (f v) 1")
    triangle_vert_ids = triangle_vert_ids.expand(
        b, -1, c
    )  # (B, faces * 3vertidsperface, C))

    vert_values = torch.zeros(
        (b, v, c), dtype=triangle_values.dtype, device=triangle_values.device
    )
    vert_values.scatter_reduce_(
        dim=1,
        index=triangle_vert_ids,
        src=triangle_values,
        reduce="mean",
        include_self=False,
    )
    return vert_values
  
  def _uv_xyz_from_verts(self, verts: Tensor):
    """calculates xyz coordinates in texture space from vertices.

    uv pixels outside of the face are set to zero.

    Args:
      vert: (B, Nverts, 3) using gnome topology

    Returns:
      uv_xyz: xyz coordinates in texture space (B, 3, uv_res, uv_res)
    """
    b = verts.shape[0]
    device = verts.device
    dtype = verts.dtype
    uv_xyz = torch.zeros(
        (self.opt.uv_res, self.opt.uv_res, b, 3), device=device, dtype=dtype
    )
    uv_xyz[self.uv_grid_mask[..., 0]] = einops.rearrange(
        self._masked_uvgridfeats_from_vertfeats(verts), "b n c -> n b c"
    ).to(uv_xyz)
    uv_xyz = einops.rearrange(uv_xyz, "h w b c -> b c h w")
    return uv_xyz

  def forward_gaussians(
      self,
      data: Dict[str, Tensor],
      dtype: torch.dtype = torch.float32,
      return_masked_gaussians: bool = False,
  ):
    """Args:

      data: A dictionary of data.
      dtype: The data type to use.

    returns:
      dict with keys:
      - xyz: (B, 1, 3, H, W)
      - rgb: (B, 1, 3, H, W)
      - scale: (B, 1, 1, H, W)
      - rotation: (B, 1, 3, H, W)
      - opacity: (B, 1, 1, H, W)
      - uv: (B, 1, 3, H, W)
    """
    b, v, _, h, w = data["uv"].shape
    data = data_util.MatchBatch(data)
    
    if self.opt.num_input_views is not None:
      device = data['uv'].device
      if not self.training:
        val_input_idcs = self.opt.val_input_idcs
        if val_input_idcs is None:
          input_idcs = torch.linspace(0, v-1, self.opt.num_input_views).to(dtype=torch.int32, device=device)
        else:
          input_idcs = torch.tensor(val_input_idcs, device=device)
      else:
        input_idcs = torch.randperm(v)[: self.opt.num_input_views].to(device)
      data = data.index_select_views(input_idcs)

    data = data.resize(self.opt.input_res)
    if self.opt.input_bg:
      bg_color = general_util.str_to_color(self.opt.input_bg)
      data = data.colorize_bg(bg_color)
    b, v, _, h, w = data["uv"].shape
    uv = data["uv"].to(torch.float16)  # computing correspondences in float16 to save memory
    downsample_for_computation = self.opt.img_patch_size // 2
    correspondence_idcs, correspondence_scores = (
        uv_util.get_img_uv_patch_correspondences(
            uv,
            self.opt.uv_res,
            uv_patch_size=self.opt.uv_patch_size,
            img_patch_size=self.opt.img_patch_size,
            n_neighbors=self.opt.n_neighbors,
            downsample_for_computation=downsample_for_computation,
            return_uv_patch_mask=False,
        )
    )
    correspondence_scores = correspondence_scores.to(dtype)
    x_img = self._get_img_tokens(data, dtype)
    x_uv, uv_xyz, uv_rgb = self._get_uv_tokens(data, dtype=dtype, correspondence_idcs=correspondence_idcs, correspondence_scores=correspondence_scores, return_uv_xyz_rgb=True)
    x_uv, x_img = self.transformer(
        uv_tokens=x_uv,
        img_tokens=x_img,
        correspondence_idcs=correspondence_idcs,
        correspondence_scores=correspondence_scores,
        nviews=v,
    )
    x_uv = self.ln_out(x_uv)
    gaussian_features = self._uv_tokens_to_gaussian_feature_textures(x_uv, uv_xyz=uv_xyz, uv_rgb=uv_rgb)


    if return_masked_gaussians:
      return gaussian_features, self.mask_gaussians(gaussian_features)
    else:
      return gaussian_features

  def forward(self, *args, func_name="compute_loss", **kwargs):
    # To support different forward functions for models wrapped by `accelerate`
    return getattr(self, func_name)(*args, **kwargs)

  def forward_mesh(
      self, data: Dict[str, Tensor], dtype: torch.dtype = torch.float32
  ) -> Tensor:
    """Args:

      data: A dictionary of data.
      dtype: The data type to use.

    Returns:
      meshes as torch tensors with shape (B, Verts, 3)
    """
    gaussian_parameters = self.forward_gaussians(data, dtype)
    verts = self.gaussians2mesh(gaussian_parameters)
    return verts

  def forward_image(self,
      data: Dict[str, Tensor],
      dtype: torch.dtype=torch.float32,
      return_masked_gaussians: bool = False):
    gaussian_parameters = self.forward_gaussians(data, dtype)
    masked_gaussian_parameters = self.mask_gaussians(gaussian_parameters)
    output_res = self.opt.output_res
    if output_res is None:
      output_res = self.opt.input_res
    data = data_util.MatchBatch(data)
    data = data.resize(output_res)
    bg_color = [1.0, 1.0, 1.0]
    images = data['image'].to(dtype)  # (B, V, 3, H, W)
    C2W = data["C2W"].to(dtype)  # (B, V, 4, 4)
    fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, V, 4)
    B, V, _, H, W = images.shape

    render_outputs = self.gs_renderer.render(
            masked_gaussian_parameters,
            C2W,
            fxfycxcy,
            height=H,
            width=W,
            bg_color=bg_color,
        )
    
    gaussian_vis_outputs = self.gs_renderer.render(
          masked_gaussian_parameters,
          C2W,
          fxfycxcy,
          height=H,
          width=W,
          render_gauss=True,
      )
    render_outputs["image_gauss"] = gaussian_vis_outputs["image"]
    if return_masked_gaussians:
      render_outputs['masked_gaussians'] = masked_gaussian_parameters
    return render_outputs
    
  @torch.no_grad()
  def get_loss_mask(self, data, device, dtype):
    B, V, _, H, W = data['image'].shape

    no_loss_mask = torch.zeros(
        (B, V, 1, H, W), dtype=torch.bool, device=device
    )  # (B, V, 1, H, W)
    for mask_name in self.opt.noloss_masks:
      no_loss_mask |= data['sg_parts'] == self.segmentation_labels[mask_name]
    # dilate the mask by k pixels
    no_loss_mask = (
        einops.rearrange(
            op_util.dilate(
                einops.rearrange(
                    no_loss_mask.float(), "b v c h w -> (b v) c h w"
                ),
                self.opt.noloss_mask_dilation,
            ),
            "(b v) c h w -> b v c h w",
            b=B,
            v=V,
        )
        > 0
    )
    loss_mask_float = 1 - no_loss_mask.to(dtype) * self.opt.noloss_mask_strength
    return loss_mask_float

  def compute_loss(
      self,
      data: Dict[str, Tensor],
      lpips_loss: lpips.LPIPS | None,
      step: int,
      dtype: torch.dtype = torch.float32,
      render_img: bool = False,
      render_uv: bool = False,
      render_gauss: bool = False,
      log_all: bool = False,
      force_white_bg: bool = False,
      return_masked_gaussians: bool = False,
  ):
    outputs = {}

    color_name = "image"

    gaussian_parameters = self.forward_gaussians(data, dtype)
    masked_gaussian_parameters = self.mask_gaussians(gaussian_parameters)
    pred_verts = self.gaussians2mesh(gaussian_parameters) * 1000.0  # in mm

    # data preprocessing for loss calculation
    output_res = self.opt.output_res
    if output_res is None:
      output_res = self.opt.input_res
    data = data_util.MatchBatch(data)
    data = data.resize(output_res)
    V = data.V
    device = data.device
    if self.opt.num_render_views is not None:
      if not self.training:
        render_idcs = torch.linspace(0, V-1, self.opt.num_render_views).to(dtype=torch.int32, device=device)
      else:
        render_idcs = torch.randperm(V)[: self.opt.num_render_views].to(device)
    data = data.index_select_views(render_idcs)

    # set background color
    bg_color = [1.0, 1.0, 1.0]
    if self.opt.output_bg and not force_white_bg:
      bg_color = general_util.str_to_color(self.opt.output_bg)
      data = data.colorize_bg(bg_color)

    images = data[color_name].to(dtype)  # (B, V, 3, H, W)
    masks = data["mask"].to(dtype)  # (B, V, 1, H, W)
    uvs = data["uv"].to(dtype)  # (B, V, 3, H, W)
    C2W = data["C2W"].to(dtype)  # (B, V, 4, 4)
    fxfycxcy = data["fxfycxcy"].to(dtype)  # (B, V, 4)
    B, V, _, H, W = images.shape
    device = images.device

    ###
    # Calculate loss weights
    ###
    loss_weights = dict()
    dataset_idx = data.get("dataset_idx", torch.zeros((B,), dtype=torch.int32))
    for k in [
        "xyz",
        "mesh_vert",
        "scale",
        "opacity",
        "render",
        'l1',
        'ssim',
        "lpips",
    ]:
      v = torch.tensor(
          getattr(self.opt, f"{k}_weight"), device=device, dtype=dtype
      )
      v = v[dataset_idx]
      loss_weights[k] = v

    # lpips warmup
    if step < self.opt.lpips_warmup_start:
      loss_weights["lpips"] = torch.zeros_like(loss_weights["lpips"])
    elif step > self.opt.lpips_warmup_end:
      pass
    else:
      loss_weights["lpips"] = (
          loss_weights["lpips"]
          * (step - self.opt.lpips_warmup_start)
          / (self.opt.lpips_warmup_end - self.opt.lpips_warmup_start)
      )

    # dropping render loss for geom only steps
    render_loss_keys = ["render", "lpips", 'l1', 'ssim']
    if step < self.opt.geom_only_steps:
      for k in render_loss_keys:
        loss_weights[k] = torch.zeros_like(loss_weights[k])

    loss_weights_binary = dict([  # used for logging
        (k, torch.ones_like(v) if log_all else (v > 0).to(v.dtype))
        for k, v in loss_weights.items()
    ])

    ###
    # Rendering (if needed)
    ###
    render_outputs = dict()
    do_render = any(  # skipping rendering if not needed
        [render_img]
        + [torch.any(loss_weights_binary[k] > 0) for k in render_loss_keys]
    )
    if do_render:
      if force_white_bg:
        render_outputs = self.gs_renderer.render(
            masked_gaussian_parameters,
            C2W,
            fxfycxcy,
            height=H,
            width=W,
            bg_color=bg_color,
        )
      else:
        render_outputs = self.gs_renderer.render(
            masked_gaussian_parameters,
            C2W,
            fxfycxcy,
            height=H,
            width=W,
            bg_color=(0.0, 0.0, 0.0),
        )
        # adding gt bg
        render_outputs["image"] = (render_outputs["image"] + (1 - render_outputs["alpha"]) * torch.tensor(bg_color, device=device, dtype=dtype).view(1, 1, 3, 1, 1))
      
      if not self.training:
        # render gaussians with white bg for visualization
        render_outputs_white = self.gs_renderer.render(
            masked_gaussian_parameters,
            C2W,
            fxfycxcy,
            height=H,
            width=W,
            bg_color=(1.0, 1.0, 1.0),
        )
        render_outputs['image_whitebg'] = render_outputs_white['image']

      for k in render_outputs.keys():
        if isinstance(render_outputs[k], Tensor):
          render_outputs[k] = render_outputs[k].to(dtype)

      # For visualization
      outputs["images/render"] = render_outputs["image"]  # (B, V, 3, H, W)
      outputs["images/normal_render"] = render_outputs["normal"] * 0.5 + 0.5  # (B, V, 3, H, W)
      outputs["images/gt"] = images
      if 'image_whitebg' in render_outputs:
        outputs['images/render_whitebg'] = render_outputs["image_whitebg"]  # (B, V, 3, H, W)

    if render_uv:
      uv_render_outputs = self.gs_renderer.render(
          masked_gaussian_parameters,
          C2W,
          fxfycxcy,
          height=H,
          width=W,
          render_uv=True,
      )
      outputs["images/uvs_render"] = uv_render_outputs["image"]
      outputs["images/uvs_gt"] = uvs

    if render_gauss:
      gaussian_vis_outputs = self.gs_renderer.render(
          masked_gaussian_parameters,
          C2W,
          fxfycxcy,
          height=H,
          width=W,
          render_gauss=True,
      )
      outputs["images/gauss_render"] = gaussian_vis_outputs["image"]

    ################################ Compute reconstruction losses/metrics ################################
    if do_render and self.opt.noloss_masks:
      loss_mask = self.get_loss_mask(data, device=images.device, dtype=images.dtype)
    else:
      loss_mask = torch.ones(
        (B, V, 1, H, W), dtype=images.dtype, device=images.device
      )  # (B, V, 1, H, W)
    
    loss = 0.0

    # gs p2p loss
    xyz = (
        einops.rearrange(
            masked_gaussian_parameters["xyz"], "b v c h w -> b (v h w) c"
        )
        * 1000.0
    )  # in mm
    gt_verts = data["verts"] * 1000.0  # in mm
    target_xyz = self._masked_uvgridfeats_from_vertfeats(gt_verts)
    p2p_loss = torch.nn.functional.mse_loss(target_xyz, xyz, reduction="none")
    if self.training:
      p2p_loss = p2p_loss * self.masked_gaussian_geometry_loss_weights[None, :, None]
    outputs["p2p_loss"] = general_util.weight_loss(
        p2p_loss, loss_weights_binary["xyz"], drop_zero_weights=True
    )  # for logging
    loss += general_util.weight_loss(
        p2p_loss, loss_weights["xyz"], drop_zero_weights=True
    )

    # mesh_vert_loss
    mesh_vert_loss = einops.rearrange(
        torch.nn.functional.mse_loss(
            einops.rearrange(pred_verts, "b v c -> v b c")[self.vertex_mask],
            einops.rearrange(gt_verts, "b v c -> v b c")[self.vertex_mask],
            reduction="none",
        ),
        "v b c -> b v c",
    )
    if self.training:
      mesh_vert_loss = mesh_vert_loss * self.geometry_loss_vertex_weights[None, :, None]
    outputs["mesh_vert_loss"] = general_util.weight_loss(
        mesh_vert_loss,
        loss_weights_binary["mesh_vert"],
        drop_zero_weights=True,
    )  # for logging
    loss += general_util.weight_loss(
        mesh_vert_loss, loss_weights["mesh_vert"], drop_zero_weights=True
    )

    # eucl distances
    gauss_eucl_distance = self.euclidean_distance(xyz, target_xyz)  # mm
    outputs["gauss_eucl_dist"] = torch.mean(gauss_eucl_distance)

    mesh_eucl_distance = self.euclidean_distance(pred_verts, gt_verts)  # mm
    outputs["mesh_eucl_dist"] = torch.mean(mesh_eucl_distance)

    masked_mesh_eucl_distance = dict()
    masked_mesh_eucl_distance["fitting_region_skin"] = torch.mean(
        self.euclidean_distance(
            einops.rearrange(pred_verts, "b v c -> v b c")[
                self.aux_vertex_mask_fitting_region_skin
            ],
            einops.rearrange(gt_verts, "b v c -> v b c")[
                self.aux_vertex_mask_fitting_region_skin
            ],
        )
    )  # mm
    for k in masked_mesh_eucl_distance.keys():
      outputs[f"mesh_eucl_dist_{k}"] = masked_mesh_eucl_distance[k]

    # scale loss
    pred_scale = (
        einops.rearrange(
            masked_gaussian_parameters["scale"], "b v c h w -> b (v h w) c"
        )
        * 1000.0
    )  # in mm
    target_scale = torch.tensor(
        [self.opt.scale_target, self.opt.scale_target, 0],
        device=xyz.device,
        dtype=xyz.dtype,
    )
    target_scale = (
        target_scale[None, None].expand_as(pred_scale) * 1000.0
    )  # in mm
    scale_loss = torch.nn.functional.mse_loss(
        pred_scale, target_scale, reduction="none"
    )
    outputs["scale_loss"] = general_util.weight_loss(
        scale_loss, loss_weights_binary["scale"], drop_zero_weights=True
    )  # for logging
    loss += general_util.weight_loss(
        scale_loss, loss_weights["scale"], drop_zero_weights=True
    )

    # opacity loss
    opacity_loss = torch.nn.functional.mse_loss(
        masked_gaussian_parameters["opacity"],
        torch.ones_like(masked_gaussian_parameters["opacity"])
        * self.opt.opacity_target,
        reduction="none",
    )
    outputs["opacity_loss"] = general_util.weight_loss(
        opacity_loss, loss_weights_binary["opacity"], drop_zero_weights=True
    )  # for logging
    loss += general_util.weight_loss(
        opacity_loss, loss_weights["opacity"], drop_zero_weights=True
    )

    # Image & Mask
    if torch.any(loss_weights_binary["render"] > 0):
      image_mse = tF.mse_loss(
          images * loss_mask,
          render_outputs["image"] * loss_mask,
          reduction="none",
      )
      outputs["image_mse"] = general_util.weight_loss(
          image_mse, loss_weights_binary["render"], drop_zero_weights=True
      )
      loss += general_util.weight_loss(
          image_mse, loss_weights["render"], drop_zero_weights=True
      )

    
    # L1
    if torch.any(loss_weights_binary["l1"] > 0):
      image_l1 = tF.l1_loss(
          images * loss_mask,
          render_outputs["image"] * loss_mask,
          reduction="none",
      )
      outputs["image_l1"] = general_util.weight_loss(
          image_l1, loss_weights_binary["l1"], drop_zero_weights=True
      )
      loss += general_util.weight_loss(image_l1, loss_weights["l1"], drop_zero_weights=True
      )

    # SSIM 
    if torch.any(loss_weights_binary["ssim"] > 0):
      image_ssim = self.ssim(
          target=einops.rearrange(images * loss_mask, 'b v c h w -> (b v) c h w'),
          preds=einops.rearrange(render_outputs["image"] * loss_mask, 'b v c h w -> (b v) c h w'),
      )
      image_ssim = einops.rearrange(image_ssim, '(b v) -> b v', b=B, v=V)
      outputs["image_ssim"] = general_util.weight_loss(
          image_ssim, loss_weights_binary["ssim"], drop_zero_weights=True
      )
      loss += general_util.weight_loss(1-image_ssim, loss_weights["ssim"], drop_zero_weights=True
      )

    # LPIPS
    if torch.any(loss_weights_binary["lpips"] > 0.0):
      assert lpips_loss is not None
      pred_img_lpips = render_outputs["image"] * loss_mask
      pred_img_lpips = einops.rearrange(
          pred_img_lpips, "b v c h w -> (b v) c h w"
      )
      pred_img_lpips = pred_img_lpips * 2.0 - 1.0
      gt_img_lpips = images * loss_mask
      gt_img_lpips = einops.rearrange(gt_img_lpips, "b v c h w -> (b v) c h w")
      gt_img_lpips = gt_img_lpips * 2.0 - 1.0
      loss_lpips = lpips_loss(pred_img_lpips, gt_img_lpips)
      loss_lpips = einops.rearrange(
          loss_lpips, "(b v) c h w -> b v (c h w)", b=B, v=V
      )
      outputs["lpips"] = general_util.weight_loss(
          loss_lpips, loss_weights_binary["lpips"], drop_zero_weights=True
      )  # for logging
      loss += general_util.weight_loss(
          loss_lpips, loss_weights["lpips"], drop_zero_weights=True
      )

    outputs["loss"] = loss

    # Metric: PSNR, SSIM and LPIPS
    if do_render:
      with torch.no_grad():
        loss_psnr = -10 * torch.log10(
            general_util.weight_loss(
                (
                    (
                        images * loss_mask
                        - render_outputs["image"].detach() * loss_mask
                    )
                    ** 2
                ),
                loss_weights_binary["render"],
                drop_zero_weights=True,
            )
        )
        outputs["psnr"] = loss_psnr
        outputs["images/render_masked"] = render_outputs["image"] * loss_mask

    if return_masked_gaussians:
      outputs["masked_gaussians"] = (
          masked_gaussian_parameters  # dict of tensors with shape (B, V, C, L, 1)
      )
    
    if self.opt.rot90:
      for k in outputs.keys():
        if k.startswith('images/'):
          outputs[k] = torch.rot90(outputs[k], k=1, dims=(-2,-1) )    

    return outputs

  def mask_gaussians(self, gaussian_parameters: dict[str, Tensor]):
    """Masks the gaussians based on the uvmap mask.

    Args:
      gaussian_parameters: A dictionary of gaussian parameters. - every
        parameters should have shape: (B, V, C, H, W)

    Returns:
      A dictionary of masked gaussian parameters.
      - every parameters will have shape: (B, V, C, L, 1)
    """

    # masking
    def mask(x):
      return einops.rearrange(
          einops.rearrange(x, "b v c h w -> h w b v c")[
              self.uv_grid_mask[..., 0]
          ],
          "l b v c -> b v c l 1",
      )

    masked_outputs = dict()
    for k in gaussian_parameters.keys():
      masked_outputs[k] = mask(gaussian_parameters[k])
    return masked_outputs

  def point_2_point_loss(
      self,
      gaussians_xyz: torch.Tensor,
      target_verts: torch.Tensor,
      reduction: str = "mean",
  ) -> torch.Tensor:
    """Computes the point-to-point loss between the predicted gaussians and the target points.

    Calculates the target xyz coordinates of the gaussians through barycentric
    interpolation from the target vertices.

    Args:
      gaussians_xyz: predicted gaussians, (B, N, 3)
      target_verts: target gnome verts, (B, V, 3)
    """

    target_gaussian_xyz = self._masked_uvgridfeats_from_vertfeats(target_verts)
    return torch.nn.functional.mse_loss(
        target_gaussian_xyz, gaussians_xyz, reduction=reduction
    )

  def euclidean_distance(
      self, pred_verts: torch.Tensor, gt_verts: torch.Tensor
  ) -> torch.Tensor:
    """Computes the euclidean distance between the predicted vertices and the target vertices.

    Args:
      pred_verts: predicted vertices, (..., 3)
      gt_verts: target vertices, (..., 3)

    Returns:
      euclidean_distance: euclidean distance between the predicted vertices and
      the
      target vertices, (...)
    """
    return torch.norm(pred_verts - gt_verts, dim=-1, p=2)

  def _masked_uvgridfeats_from_vertfeats(
      self, vertex_features: torch.Tensor
  ) -> torch.Tensor:
    """Barycentric interpolation of gnome vertex features on masked uv grid points.

    Args:
      vertex_features: gnome vertex features, (B, P, C).

    Returns:
      Masked uv grid features, (B, N, C).
    """
    vertex_features = einops.rearrange(vertex_features, "B P C -> P C B")
    vertex_features = vertex_features[
        self.uv_grid_vert_ids_filtered
    ]  # (N_filtered, 3[bary], 3[xyz], B)
    bary_coords = self.uv_grid_bary_filtered[
        ..., None, None
    ]  # (N_filtered, 3[bary], 1, 1)
    uv_grid_vertex_features = torch.sum(
        vertex_features * bary_coords, axis=1
    )  # (N_filtered, 3[xyz], B)
    uv_grid_vertex_features = einops.rearrange(
        uv_grid_vertex_features, "N C B -> B N C"
    )
    return uv_grid_vertex_features

  def _register_uvcaches(
      self,
      uv_res: int,
      use_vertex_group_only: str | None = None,
  ):
    """registers the barycentric coordinates of the uv grid points wrt GNome

    Args:
      uv_res: resolution of the uv grid.
      use_vertex_group_only: vertex group to use for filtering the mesh.

    Registers:
      - template_triangles: gnome triangles, (T, 3)
      - uv_grid_coords: uv coordinates of the uv grid, (uv_res, uv_res, 2)
        [0...1], origin is on bottom left, bottom left pixel center is (0, 0)
      - uv_grid_mask: mask of the uv grid, (uv_res, uv_res, 1)
      - uv_grid_vert_ids_filtered: triangle vertex ids of the uv grid points,
        (N_filtered, 3)
      - uv_grid_bary_filtered: barycentric coordinates of the uv grid points,
        (N_filtered, 3)
      - uv_grid_bary_au_filtered: barycentric coordinates of the uv grid points
    """

    # loading mesh
    template_path = self.opt.template_mesh_path
    if not file_util.exists(template_path):
      template_path = file_util.get_resource_filename(template_path)
    mesh_info = mesh_util.load_obj(str(template_path))
    template_verts = mesh_info['v']
    template_triangles = mesh_info['vi']
    template_triangle_uvs = mesh_info['vt'][mesh_info['vti']]

    # vertex group weights
    with file_util.open_file(self.opt.template_mesh_vertex_groups_path, 'r') as f:
      vertex_groups = json.load(f) 
    self.template_vertex_group_names = sorted(vertex_groups.keys())
    G = len(self.template_vertex_group_names)
    V = len(template_verts)
    template_vertex_group_weights = np.zeros((G, V), dtype=np.float32)
    for i in range(G):
      template_vertex_group_weights[i][vertex_groups[self.template_vertex_group_names[i]]] = 1.

    face_mask, vertex_mask = mesh_util.filter_mesh_by_vertex_group(
        template_triangles,
        template_vertex_group_weights,
        self.template_vertex_group_names,
        use_vertex_group_only,
    )
    template_triangles_orig = template_triangles.copy()
    template_triangle_uvs_orig = (
        template_triangle_uvs.copy()
    )

    template_triangles = template_triangles[face_mask]
    template_triangle_uvs = template_triangle_uvs[face_mask]

    for aux_mask_name in ["fitting_region_skin"]:
      aux_face_mask, aux_vertex_mask = mesh_util.filter_mesh_by_vertex_group(
          template_triangles,
          template_vertex_group_weights,
          self.template_vertex_group_names,
          aux_mask_name,
      )
      self.register_buffer(
          f"aux_face_mask_{aux_mask_name}",
          torch.from_numpy(aux_face_mask),
          persistent=False,
      )
      self.register_buffer(
          f"aux_vertex_mask_{aux_mask_name}",
          torch.from_numpy(aux_vertex_mask),
          persistent=False,
      )

    self.register_buffer(
        "template_vertices", torch.from_numpy(template_verts), persistent=True
    )

    self.register_buffer(
        "template_triangles", torch.from_numpy(template_triangles), persistent=True
    )
    self.register_buffer(
        "template_triangle_uvs",
        torch.from_numpy(template_triangle_uvs.astype(np.float32)),
        persistent=True,
    )
    self.register_buffer(
        "num_verts",
        torch.tensor(template_vertex_group_weights.shape[1]),
        persistent=True,
    )
    self.register_buffer(
        "template_vertex_group_weights",
        torch.from_numpy(template_vertex_group_weights),
        persistent=False,
    )
    self.register_buffer(
        "template_triangles_orig",
        torch.from_numpy(template_triangles_orig),
        persistent=True,
    )
    self.register_buffer(
        "template_triangle_uvs_orig",
        torch.from_numpy(template_triangle_uvs_orig),
        persistent=True,
    )
    self.register_buffer(
        "face_mask", torch.from_numpy(face_mask), persistent=True
    )
    self.register_buffer(
        "vertex_mask", torch.from_numpy(vertex_mask), persistent=True
    )

    # uv grid
    (
        uv_coords,
        uv_grid_bary,  # (uv_res, uv_res, 3)
        uv_grid_triangle_id,  # (uv_res, uv_res, 1)
        uv_grid_mask,  # (uv_res, uv_res, 1)
        uv_grid_bary_au,  # (uv_res, uv_res, 3)
        uv_grid_bary_av,  # (uv_res, uv_res, 3)
    ) = uv_util.get_uvgrid_barycentric_coords(template_triangle_uvs, uv_res)
    self.register_buffer(
        "uv_grid_coords",
        torch.from_numpy(uv_coords.astype(np.float32)),
        persistent=True,
    )

    self.register_buffer(
        "uv_grid_mask", torch.from_numpy(uv_grid_mask), persistent=True
    )

    self.register_buffer(
        "uv_grid_bary",
        torch.from_numpy(uv_grid_bary.astype(np.float32)),
        persistent=True,
    )
    self.register_buffer(
        "uv_grid_triangle_id",
        torch.from_numpy(uv_grid_triangle_id),
        persistent=True,
    )
    self.register_buffer(
        "uv_grid_bary_au",
        torch.from_numpy(uv_grid_bary_au.astype(np.float32)),
        persistent=True,
    )
    self.register_buffer(
        "uv_grid_bary_av",
        torch.from_numpy(uv_grid_bary_av.astype(np.float32)),
        persistent=True,
    )

    geometry_loss_weights_by_part = self.opt.geometry_loss_weights_by_part
    geometry_loss_vertex_weights = np.zeros((len(self.template_vertices),), dtype=np.float32) + geometry_loss_weights_by_part["__default__"]
    for k in sorted(geometry_loss_weights_by_part.keys()):
      if k == '__default__':
        continue
      else:
        geometry_loss_vertex_weights[vertex_groups[k]] = geometry_loss_weights_by_part[k]
    self.register_buffer("geometry_loss_vertex_weights",
                         torch.from_numpy(geometry_loss_vertex_weights.astype(np.float32)),
                         persistent=False,
                         ) 
  @property
  def masked_gaussian_geometry_loss_weights(self) -> torch.Tensor:  # recompute this every time so that changes in loss weights are reflected, but use persistent uv_mask
    if not hasattr(self, '_masked_gaussian_geometry_loss_weights'):  
      _masked_gaussian_geometry_loss_weights = self._masked_uvgridfeats_from_vertfeats(
        einops.rearrange(self.geometry_loss_vertex_weights, 'v -> 1 v 1')
      )[0, ..., 0]  # (N_masked,)
      self.register_buffer('_masked_gaussian_geometry_loss_weights', _masked_gaussian_geometry_loss_weights, persistent=False)
    return self._masked_gaussian_geometry_loss_weights


  @property
  def uv_grid_vert_ids_filtered(self) -> torch.Tensor:
    return self.template_triangles[
        self.uv_grid_triangle_id[self.uv_grid_mask]
    ]  # (N_filtered, 3)

  @property
  def uv_grid_bary_filtered(self) -> torch.Tensor:
    return self.uv_grid_bary[self.uv_grid_mask[..., 0]]  # (N_filtered, 3))

  @property
  def uv_grid_bary_au_filtered(self) -> torch.Tensor:
    return self.uv_grid_bary_au[self.uv_grid_mask[..., 0]]  # (N_filtered, 3))

  @property
  def uv_grid_bary_av_filtered(self) -> torch.Tensor:
    return self.uv_grid_bary_av[self.uv_grid_mask[..., 0]]  # (N_filtered, 3))

  @torch.amp.autocast("cuda", torch.float32)
  def _uv_tokens_to_gaussian_feature_textures(self, x: Tensor, uv_xyz=None, uv_rgb=None):
    """Converts uv tokens to gaussian feature textures.

    Args:
      x: uv tokens, (B, N, D)

    Returns:
      dict with keys:
      - xyz: (B, 1, 3, H, W)
      - rgb: (B, 1, 3, H, W)
      - scale: (B, 1, 1, H, W)
      - rotation: (B, 1, 3, H, W)
      - opacity: (B, 1, 1, H, W)
      - uv: (B, 1, 3, H, W)
    """

    def _reshape_feature(features: Tensor):
      features = einops.rearrange(
          features,
          "b (v h w) d -> (b v) (h w) d",
          v=1,  # V_in,
          h=self.uv_inter_res,
      )
      features = op_util.unpatchify(
          features,
          self.opt.uv_patch_size,
          self.uv_inter_res,
      )
      features = einops.rearrange(
          features,
          "(b v) c h w -> b v c h w",
          v=1,  # V_in,
      )  # (B, V_in, `dim`, H, W)
      return features

    xyz = _reshape_feature(self.out_xyz(x))
    rgb = _reshape_feature(self.out_rgb(x))
    scale = _reshape_feature(self.out_scale(x))
    rotation = _reshape_feature(self.out_rotation(x))
    opacity = _reshape_feature(self.out_opacity(x))
    uv = (
        torch.cat(
            [
                self.uv_grid_coords,
                torch.ones_like(self.uv_grid_coords[:, :, :1]),
            ],
            dim=-1,
        )
        .permute(2, 0, 1)[None, None]
        .expand_as(xyz)
    )

    xyz = torch.sigmoid(xyz) * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    rgb = torch.sigmoid(rgb)  # [0, 1]
    scale = self.scale_activation(
        torch.sigmoid(scale)
    )  # [0, 1] -> [smin, smax]
    rotation = tF.normalize(rotation, p=2, dim=2)  # L2 normalize [-1, 1]
    opacity = torch.sigmoid(
        opacity - 2.0
    )  # [0, 1]; `-2.` cf. GS-LRM Appendix A.4
    uv = uv  # [0, 1]

    if self.skip_connection:
      xyz = xyz + einops.rearrange(uv_xyz, 'b c h w -> b 1 c h w')
      rgb = (rgb * 2 - 1) + einops.rearrange(uv_rgb, 'b c h w -> b 1 c h w')  # when using skip connections, color prediction must also be capable of producing negative values
    return {
        "xyz": xyz,
        "rgb": rgb,
        "scale": scale,
        "rotation": rotation,
        "opacity": opacity,
        "uv": uv,
    }

