import torch
from torch import nn
from torch.nn import functional
import pudb
def activation_layer(*, type: str | None) -> nn.Module:
  match type:
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
      raise ValueError(f'{type}')


def normalization_layer(*, d: int, type: str | None) -> nn.Module:
  match type:
    case None | 'none':
      return nn.Identity()
    case 'bn':
      return nn.BatchNorm2d(num_features=d)
    case 'in':
      return nn.InstanceNorm2d(num_features=d)
    case _:
      raise ValueError(f'{type}')

def optional_resize(
        tensor: torch.Tensor, target_shape: torch.Tensor
    ) -> torch.Tensor:
      if tensor.shape[2:] != target_shape:
        return functional.interpolate(tensor, size=target_shape)
      else:
        return tensor

class ResidualBlock(nn.Module):

  def __init__(
      self,
      c_in: int,
      c_out: int,
      downsample: bool = False,
      normalization: str | None = 'in',
      activation: str | None = 'relu',
  ):
    super().__init__()

    self.conv1 = ConvBlock(
        c_in=c_in,
        c_out=c_out,
        downsample=downsample,
        normalization=normalization,
        activation=activation,
    )
    self.conv2 = ConvBlock(
        c_in=c_out,
        c_out=c_out,
        downsample=False,
        normalization=normalization,
        activation=None,
    )
    self.skip = Skip(
        c_in=c_in,
        c_out=c_out,
        downsample=downsample,
        normalization=normalization,
    )
    self.activation = activation_layer(type=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    skip = self.skip(x)
    x = self.conv1(x)
    x = self.conv2(x)
    x = x + skip
    return self.activation(x)


class ConvBlock(nn.Module):

  def __init__(
      self,
      c_in: int,
      c_out: int,
      activation: str | None = 'relu',
      normalization: str | None = 'in',
      downsample: bool = False,
  ):
    super().__init__()

    s = 2 if downsample else 1
    p = 1 if downsample else 'same'
    self.conv = nn.Conv2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=3,
        stride=s,
        padding=p,
        bias=False,
    )
    self.norm = normalization_layer(d=c_out, type=normalization)
    self.activation = activation_layer(type=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.norm(x)
    return self.activation(x)


class Skip(nn.Module):

  def __init__(
      self,
      c_in: int,
      c_out: int,
      normalization: str = 'in',
      downsample: bool = False,
  ):
    super().__init__()

    k = 3 if downsample else 1
    s = 2 if downsample else 1
    p = 1 if downsample else 'same'
    self.norm = normalization_layer(d=c_out, type=normalization)
    self.conv = nn.Conv2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=k,
        stride=s,
        padding=p,
        bias=False,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.norm(x)
    return x


class UpBlock(nn.Module):

  def __init__(
      self,
      c_in: int,
      c_out: int,
      normalization: str | None = 'in',
      activation: str | None = 'relu',
  ):
    super().__init__()

    self.up_conv = nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=2,
        stride=2,
    )
    self.norm = normalization_layer(d=c_out, type=normalization)
    self.activation = activation_layer(type=activation)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.up_conv(x)
    x = self.norm(x)
    return self.activation(x)

class DownBlock(nn.Module):
  def __init__(self, c_in: int, c_out: int, downsample: bool = True, normalization: str = 'in', activation: str = 'relu'):
    super().__init__()
    self.block1 = ResidualBlock(
              c_in=c_in,
              c_out=c_out,
              downsample=downsample,
              normalization=normalization,
              activation=activation,
          )
    self.block2 = ResidualBlock(
              c_in=c_out,
              c_out=c_out,
              downsample=False,
              normalization=normalization,
              activation=activation,
          )
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.block1(x)
    x = self.block2(x)
    return x


class CustomFeatureExtractorNet(nn.Module):

  def __init__(
      self,
      c_in: int,
      c_out: int,
      s_hidden: int = 2,
      normalization: str = 'in',
      activation: str = 'relu',
  ):
    """Initializes a image-to-image feature extractor network.

    image (B, 3, H, W) -> features (B, H, W, out_channels)
    """
    super().__init__()

    self.input_conv = nn.Sequential(
        nn.Conv2d(
            in_channels=c_in,
            out_channels=32,
            kernel_size=1,
            stride=1,
            bias=False,
        ),
        normalization_layer(d=32, type=normalization),
        activation_layer(type=activation),
    )
    self.input_pool = nn.Sequential(
        nn.AvgPool2d(
            kernel_size=1, stride=1
        ),
        nn.Conv2d(
            in_channels=c_in,
            out_channels=32,
            kernel_size=1,
            stride=1,
        ),
        normalization_layer(d=32, type=normalization),
        activation_layer(type=activation),
    )

    self.enc1 = DownBlock(c_in=32, c_out=64, normalization=normalization, activation=activation)
    self.enc2 = DownBlock(c_in=64, c_out=128, normalization=normalization, activation=activation)
    self.enc3 = DownBlock(c_in=128, c_out=256, normalization=normalization, activation=activation)
    self.enc4 = DownBlock(c_in=256, c_out=512, normalization=normalization, activation=activation)

    self.dec1 = UpBlock(c_in=512, c_out=256, normalization=normalization, activation=activation)
    self.dec2 = UpBlock(c_in=256, c_out=128, normalization=normalization, activation=activation)
    self.dec3 = UpBlock(c_in=128, c_out=64, normalization=normalization, activation=activation)
    self.dec4 = UpBlock(c_in=64, c_out=32, normalization=normalization, activation=activation)

    self.skip1 = Skip(c_in=32, c_out=32, normalization=normalization)
    self.skip2 = Skip(c_in=64, c_out=64, normalization=normalization)
    self.skip3 = Skip(c_in=128, c_out=128, normalization=normalization)
    self.skip4 = Skip(c_in=256, c_out=256, normalization=normalization)

    self.out = nn.Sequential(
        ConvBlock(
            c_in=32,
            c_out=c_out,
            downsample=False,
            normalization=normalization,
        ),
        nn.Conv2d(
            in_channels=c_out, out_channels=c_out, kernel_size=1
        ),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    x = self.input_conv(x) + self.input_pool(x)
    skip1 = self.skip1(x)
    x = self.enc1(x)

    skip2 = self.skip2(x)
    x = self.enc2(x)

    skip3 = self.skip3(x)
    x = self.enc3(x)

    skip4 = self.skip4(x)
    x = self.enc4(x)

    x = self.dec1(x)
    x = skip4 + optional_resize(x, skip4.shape[2:])

    x = self.dec2(x)
    x = skip3 + optional_resize(x, skip3.shape[2:])

    x = self.dec3(x)
    x = skip2 + optional_resize(x, skip2.shape[2:])

    x = self.dec4(x)
    x = skip1 + optional_resize(x, skip1.shape[2:])

    x = self.out(x)
    return x


if __name__ == "__main__":
  torch.manual_seed(1234)
  model = CustomFeatureExtractorNet(
    c_in=3,
    c_out=128,
    normalization="in",
    activation="relu",
  )
  model.eval()
  print(model.forward(torch.randn(1, 3, 256, 256)))