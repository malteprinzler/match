from typing import Dict, Optional, Tuple
import einops
from gsplat.rendering import rasterization, rasterization_2dgs
import pudb
from match.options import Options
import torch
from torch import Tensor
import torch.nn.functional as tF


class GSplatRenderer:
  """Wrapper around the gsplat renderer."""

  def __init__(self, twoDgs=False, overrendering_factor=1):
    """initializes the GSplatRenderer.

    Args:
      twoDgs: Whether to use 2DGS rendering.
      overrendering_factor: The overrendering factor to use, i.e. rendering
        splats at overrendering_factor x higher resolution and downsampling
        afterwards with antialiased bilinear interpolation.
    """
    self.twoDgs = twoDgs
    self.rasterization_fn = rasterization_2dgs if twoDgs else rasterization
    self.overrendering_factor = overrendering_factor

  @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
  def render(
      self,
      model_outputs: Dict[str, Tensor],
      C2W: Tensor,
      fxfycxcy: Tensor,
      height: Optional[float] = None,
      width: Optional[float] = None,
      bg_color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
      opacity_threshold: float = 0.0,
      in_image_format: bool = True,
      render_uv: bool = False,
      render_gauss: bool = False,
  ):
    """Renders a GaussianMesh from a set of Gaussian parameters.

    masked_gaussian_parameters: A dictionary of Gaussian parameters.
     - rgb: The color of the Gaussian.
     - scale: The scale of the Gaussian.
     - rotation: The rotation of the Gaussian.
     - opacity: The opacity of the Gaussian.
     - uv: The uv coordinate of the Gaussian.
     - xyz: The xyz coordinate of the Gaussian.
     All with shape (B, N, C) or (B, V, C, H, W) if in_image_format is True.


    Args:
      masked_gaussian_parameters: A dictionary of Gaussian parameters.
      C2W: The camera to world transform. (B, V, 4, 4)
      fxfycxcy: The camera's focal length, and principal point normalized to
        0...1, (B, V, 4)
      height: The height of the image.
      width: The width of the image.

    Returns:
      A dictionary of rendered images.
        - image: The rendered image. (B, V, 3, H, W)
        - alpha: The alpha of the rendered image. (B, V, 1, H, W)
        - normal: [BUGGY] The normal of the rendered image. (B, V, 3, H, W)
    """
    out_height = height
    out_width = width
    v = C2W.shape[1]
    device = C2W.device
    dtype = C2W.dtype

    height = int(height * self.overrendering_factor)
    width = int(width * self.overrendering_factor)

    if not in_image_format:
      assert height is not None and width is not None
      assert "xyz" in model_outputs  # depth must be in image format

    if render_uv:
      bg_color = (0, 0, 0.0)
    elif render_gauss:
      bg_color = (1, 1, 1.0)
    bg_color = torch.tensor(list(bg_color), dtype=dtype, device=device)[
        None
    ].expand((v, 3))

    rgb, scale, rotation, opacity, uv, xyz = (
        model_outputs["rgb"],
        model_outputs["scale"],
        model_outputs["rotation"],
        model_outputs["opacity"],
        model_outputs["uv"],
        model_outputs["xyz"],
    )

    # Rendering resolution could be different from input resolution
    H = height if height is not None else rgb.shape[-2]
    W = width if width is not None else rgb.shape[-1]

    # Reshape for rendering
    if in_image_format:
      rgb = einops.rearrange(rgb, "b v c h w -> b (v h w) c")
      scale = einops.rearrange(scale, "b v c h w -> b (v h w) c")
      rotation = einops.rearrange(rotation, "b v c h w -> b (v h w) c")
      opacity = einops.rearrange(opacity, "b v c h w -> b (v h w) c")
      uv = einops.rearrange(uv, "b v c h w -> b (v h w) c")

    # setting gs colors depending on rendering mode
    if render_uv:
      rgb = uv
    elif render_gauss:
      randgen = torch.Generator(device=rgb.device).manual_seed(0)
      rgb = torch.rand(
          rgb[:1].shape, device=rgb.device, dtype=dtype, generator=randgen
      ).expand_as(rgb)
    else:
      pass

    # Prepare XYZ for rendering
    xyz = xyz + model_outputs.get("offset", torch.zeros_like(xyz))
    if in_image_format:
      xyz = einops.rearrange(xyz, "b v c h w -> b (v h w) c")

    # Filter by opacity
    opacity = (opacity > opacity_threshold) * opacity

    B, V = C2W.shape[:2]  # `HR`/`WR` meight be different from `H`/`W`
    images = torch.zeros(B, V, 3, H, W, dtype=dtype, device=device)
    alphas = torch.zeros(B, V, 1, H, W, dtype=dtype, device=device)
    normals = torch.zeros(B, V, 3, H, W, dtype=dtype, device=device)

    # perparing camera parameters
    Ks = torch.zeros(B, V, 3, 3, dtype=dtype, device=device)
    Ks[:, :, 0, 0] = fxfycxcy[:, :, 0] * width
    Ks[:, :, 1, 1] = fxfycxcy[:, :, 1] * height
    Ks[:, :, 0, 2] = fxfycxcy[:, :, 2] * width
    Ks[:, :, 1, 2] = fxfycxcy[:, :, 3] * height
    Ks[:, :, 2, 2] = 1

    W2C = invert_c2w(C2W)

    for i in range(B):
      raster_outputs = self.rasterization_fn(
          means=xyz[i],
          quats=rotation[i],
          scales=scale[i],
          opacities=opacity[i, ..., 0],
          colors=rgb[i],
          viewmats=W2C[i],
          Ks=Ks[i],
          width=width,
          height=height,
          backgrounds=bg_color,
      )[:3]
      images[i] = einops.rearrange(raster_outputs[0], "v h w c -> v c h w")
      alphas[i] = einops.rearrange(raster_outputs[1], "v h w c -> v c h w")
      if self.twoDgs:
        normals[i] = einops.rearrange(raster_outputs[2], "v h w c -> v c h w")

    return_dict = {
        "image": images,
        "alpha": alphas,
        "normal": normals,
    }

    for k in return_dict.keys():
      if k != "pc":
        return_dict[k] = tF.interpolate(
            einops.rearrange(return_dict[k], "b v c h w -> (b v) c h w"),
            size=(out_height, out_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        return_dict[k] = einops.rearrange(
            return_dict[k], "(b v) c h w -> b v c h w", v=V, b=B
        )
    return return_dict


def invert_c2w(c2w: Tensor) -> Tensor:
  """Inverts a camera to world transform.

  Args:
    c2w: The camera to world transform. (B, V, 4, 4)

  Returns:
    The inverted camera to world transform. (B, V, 4, 4)
  """
  w2c = torch.zeros_like(c2w)
  w2c[..., 3, 3] = 1
  w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
  w2c[..., :3, -1:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, -1:]
  return w2c
