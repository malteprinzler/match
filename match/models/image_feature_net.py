"""Extracts feature maps from images.

Dimension notations:
B: Batch size.
H: Image height.
W: Image width.
F: Feature dimension.
"""

import enum
import math
import gin
import pudb
import torch
from torch import nn
from torch.nn import functional


def _norm(*, num_features: int, norm: str | None) -> nn.Module:
  """Returns normalization layer."""
  match norm:
    case None | 'none':
      return nn.Identity()
    case 'bn':
      return nn.BatchNorm2d(num_features=num_features)
    case 'in':
      return nn.InstanceNorm2d(num_features=num_features)
    case _:
      raise ValueError(f'Unsupported normalization type: {norm}')


def _activation(*, activation: str | None) -> nn.Module:
  """Returns activation layer."""
  match activation:
    case None | 'none':
      return nn.Identity()
    case 'relu':
      return nn.ReLU(inplace=True)
    case 'elu':
      return nn.ELU(inplace=True)
    case 'leaky_relu':
      return nn.LeakyReLU(inplace=True)
    case 'sigmoid':
      return nn.Sigmoid()
    case _:
      raise ValueError(f'Unsupported activation type: {activation}')


def _resize_if_needed(
    tensor: torch.Tensor, target_shape: tuple[int, int]
) -> torch.Tensor:
  if tensor.shape[2:] != target_shape:
    return functional.interpolate(
        tensor,
        size=target_shape,
        mode='bilinear',
        align_corners=True,
    )
  else:
    return tensor


class _SkipBlock(nn.Module):
  """Linear convolutional layer with layer norm used for skip connections."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_channels: int,
      downsample: bool = False,
      norm: str | None = 'in',
  ):
    super().__init__()

    kernel_size = 3 if downsample else 1
    stride = 2 if downsample else 1
    padding = 1 if downsample else 'same'
    self._conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False,
    )
    self._norm = _norm(num_features=out_channels, norm=norm)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._conv(x)
    x = self._norm(x)
    return x


class _ConvBlock(nn.Module):
  """Convolutional block with norm and activation."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_channels: int,
      downsample: bool = False,
      norm: str | None = 'in',
      activation: str | None = 'relu',
  ):
    super().__init__()

    stride = 2 if downsample else 1
    padding = 1 if downsample else 'same'
    self._conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )
    self._norm = _norm(num_features=out_channels, norm=norm)
    self._activation = _activation(activation=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._conv(x)
    x = self._norm(x)
    return self._activation(x)


class _ResBlock(nn.Module):
  """Convolutional block with skip connection."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_channels: int,
      downsample: bool = False,
      norm: str | None = 'in',
      activation: str | None = 'relu',
      use_squeeze_and_excitation: bool = True,
  ):
    super().__init__()

    self._conv1 = _ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        downsample=downsample,
        norm=norm,
        activation=activation,
    )
    self._conv2 = _ConvBlock(
        in_channels=out_channels,
        out_channels=out_channels,
        downsample=False,
        norm=norm,
        activation=None,
    )
    self._skip = _SkipBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        downsample=downsample,
        norm=norm,
    )
    self._activation = _activation(activation=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    skip = self._skip(x)
    x = self._conv1(x)
    x = self._conv2(x)
    x = x + skip
    return self._activation(x)


class _UpsampleBlock(nn.Module):
  """Upsampling block by a factor of two."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_channels: int,
      norm: str | None = 'in',
      activation: str | None = 'relu',
  ):
    super().__init__()

    self._up_conv = nn.ConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=2,
        stride=2,
    )
    self._norm = _norm(num_features=out_channels, norm=norm)
    self._activation = _activation(activation=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self._up_conv(x)
    x = self._norm(x)
    return self._activation(x)


class LoRALinear(nn.Module):
  """Linear layer with LoRA."""

  def __init__(self, *, linear_layer: nn.Linear, rank: int):
    super().__init__()

    self._linear_layer = linear_layer
    self._rank = rank

    if self._rank > 0:
      self._lora_layers = nn.Sequential(
          nn.Linear(linear_layer.in_features, self._rank, bias=False),
          nn.Linear(self._rank, linear_layer.out_features, bias=False),
      )
      nn.init.kaiming_uniform_(self._lora_layers[0].weight, a=math.sqrt(5))
      nn.init.zeros_(self._lora_layers[1].weight)
    else:
      self._lora_layers = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self._linear_layer(x) + self._lora_layers(x)


def _apply_lora(module: nn.Module, rank: int):
  for name, child in module.named_children():
    if isinstance(child, nn.Linear):
      setattr(module, name, LoRALinear(linear_layer=child, rank=rank))
    else:
      _apply_lora(child, rank)


class ImageFeatureNet(nn.Module):
  """Extraction of feature maps from images with the same height and width."""

  def __init__(
      self,
      *,
      in_channels: int,
      out_channels: int,
      hidden_layers_scale: int = 2,
      norm: str = 'in',
      activation: str = 'relu',
      image_downscale_factor: int = 1,
      use_squeeze_and_excitation: bool = False,
      **unused_kwargs,
  ):
    """Initializes a image-to-image feature extractor network.

    Given a batched image of shape (B, H, W, 3), the feature extractor predicts
    a feature map of shape (B, H, W, out_channels). Note that this differs from
    the (B, C, H, W) format, commonly used in PyTorch.

    Args:
      in_channels: The number of channels of the input images.
      out_channels: The number of channels of the predicted feature maps.
      hidden_layers_scale: A factor to cale the size of internal layers.
      norm: The layer norm.
      activation: The used activation function.
      image_downscale_factor: A scaling factor applied to downscale the input
        images. The value must be a power of 2.
      use_squeeze_and_excitation: If True, sequeeze-and-excitation blocks are
        used in the encoder residual layers.
      **unused_kwargs: Optional additional parameters not used in this image
        feature extractor.
    """
    super().__init__()

    def _encoder_block(
        in_channels: int, out_channels: int, downsample: bool = True
    ) -> nn.Module:
      return nn.Sequential(
          _ResBlock(
              in_channels=in_channels,
              out_channels=out_channels,
              downsample=downsample,
              norm=norm,
              activation=activation,
          ),
          _ResBlock(
              in_channels=out_channels,
              out_channels=out_channels,
              downsample=False,
              norm=norm,
              activation=activation,
          ),
      )

    def _decoder_block(in_channels: int, out_channels: int) -> nn.Module:
      return _UpsampleBlock(
          in_channels=in_channels,
          out_channels=out_channels,
          norm=norm,
          activation=activation,
      )

    if math.log2(image_downscale_factor) % 1 != 0:
      raise ValueError(
          f'The image_downscale_factor {image_downscale_factor} must be a power'
          ' of two.'
      )

    self._stem_conv = nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=image_downscale_factor,
            stride=image_downscale_factor,
            bias=False,
        ),
        _norm(num_features=32, norm=norm),
        _activation(activation=activation),
    )
    self._stem_pool = nn.Sequential(
        nn.AvgPool2d(
            kernel_size=image_downscale_factor, stride=image_downscale_factor
        ),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=1,
            stride=1,
        ),
        _norm(num_features=32, norm=norm),
        _activation(activation=activation),
    )

    self._enc_block1 = _encoder_block(in_channels=32, out_channels=64)
    self._enc_block2 = _encoder_block(in_channels=64, out_channels=128)
    self._enc_block3 = _encoder_block(in_channels=128, out_channels=256)
    self._enc_block4 = _encoder_block(in_channels=256, out_channels=512)

    self._decoder_block1 = _decoder_block(in_channels=512, out_channels=256)
    self._decoder_block2 = _decoder_block(in_channels=256, out_channels=128)
    self._decoder_block3 = _decoder_block(in_channels=128, out_channels=64)
    self._decoder_block4 = _decoder_block(in_channels=64, out_channels=32)

    self._skip1 = _SkipBlock(in_channels=32, out_channels=32, norm=norm)
    self._skip2 = _SkipBlock(in_channels=64, out_channels=64, norm=norm)
    self._skip3 = _SkipBlock(in_channels=128, out_channels=128, norm=norm)
    self._skip4 = _SkipBlock(in_channels=256, out_channels=256, norm=norm)

    self._final_layer = nn.Sequential(
        _ConvBlock(
            in_channels=32,
            out_channels=out_channels,
            downsample=False,
            norm=norm,
        ),
        nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=1
        ),
    )

  def forward(self, images: torch.Tensor) -> torch.Tensor:
    """Predicts feature maps from images.

    Args:
      images: The batched images, (B, H, W, 3).

    Returns:
      Feature map of size (B, H, W, F).
    """

    x = images.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    x_conv = self._stem_conv(x)
    x_pool = self._stem_pool(x)
    x = x_conv + x_pool

    skip1 = self._skip1(x)
    x = self._enc_block1(x)

    skip2 = self._skip2(x)
    x = self._enc_block2(x)

    skip3 = self._skip3(x)
    x = self._enc_block3(x)

    skip4 = self._skip4(x)
    x = self._enc_block4(x)

    x = self._decoder_block1(x)
    x = skip4 + _resize_if_needed(x, skip4.shape[2:])

    x = self._decoder_block2(x)
    x = skip3 + _resize_if_needed(x, skip3.shape[2:])

    x = self._decoder_block3(x)
    x = skip2 + _resize_if_needed(x, skip2.shape[2:])

    x = self._decoder_block4(x)
    x = skip1 + _resize_if_needed(x, skip1.shape[2:])

    x = self._final_layer(x)
    return x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)




@gin.constants_from_enum
@enum.unique
class ImageFeatureNetType(enum.Enum):
  IMAGE_FEATURE_NET = enum.auto()

  def __str__(self):
    return self.name


_IMAGE_FEATURE_NET_MAP = {
    ImageFeatureNetType.IMAGE_FEATURE_NET: ImageFeatureNet
}


class ImageFeatureNetContainer:
  """Container to handle different image feature prediction networks."""

  def __init__(self, image_feature_net_type: ImageFeatureNetType):
    self._image_feature_net_type = image_feature_net_type
    self._model = _IMAGE_FEATURE_NET_MAP[image_feature_net_type]

  def model(self, **kwargs) -> ImageFeatureNet:
    return self._model(**kwargs)
