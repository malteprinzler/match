# CC=gcc-11 CXX=g++-11 CUDA_VISIBLE_DEVICES=0, python vr/perception/multiview_mesh_prediction/gtempeh_standalone/src/models/gsplat_renderer/gsplat_renderer_cli.py
from gsplat.rendering import rasterization, rasterization_2dgs
import pudb
import torch


def main():
  device = torch.device("cuda:0")
  # N = 1000
  # means = torch.randn(1, N, 3, device=device)
  # quats = torch.randn(1, N, 4, device=device)
  # scales = torch.rand(1, N, 3, device=device)
  # opacities = torch.rand(1, N, device=device)
  # colors = torch.rand(1, N, 3, device=device)
  # Ks = torch.tensor(
  #     [[
  #         [
  #             [1.9605e03, 0.0000e00, 3.8400e02],
  #             [0.0000e00, 1.9610e03, 2.5600e02],
  #             [0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [1.9680e03, 0.0000e00, 3.8400e02],
  #             [0.0000e00, 1.9670e03, 2.5600e02],
  #             [0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [1.9680e03, 0.0000e00, 3.8400e02],
  #             [0.0000e00, 1.9680e03, 2.5600e02],
  #             [0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [1.9680e03, 0.0000e00, 3.8400e02],
  #             [0.0000e00, 1.9680e03, 2.5600e02],
  #             [0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #     ]],
  #     device=device,
  # )
  # viewmats = torch.tensor(
  #     [[
  #         [
  #             [-4.5013e-02, 9.9463e-01, 9.4543e-02, 4.5880e-03],
  #             [-4.8370e-02, -9.6741e-02, 9.9414e-01, -7.0310e-02],
  #             [9.9805e-01, 4.0192e-02, 5.2460e-02, 1.2293e00],
  #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [2.3291e-01, 9.4189e-01, 2.4146e-01, -1.5000e-03],
  #             [-5.2100e-01, -8.8806e-02, 8.4863e-01, -6.0937e-02],
  #             [8.2129e-01, -3.2349e-01, 4.7021e-01, 1.0341e00],
  #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [-2.1887e-01, 9.7559e-01, 6.5804e-05, -1.8565e-02],
  #             [-5.0293e-01, -1.1292e-01, 8.5693e-01, -6.0447e-02],
  #             [8.3594e-01, 1.8750e-01, 5.1562e-01, 9.5959e-01],
  #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #         [
  #             [-2.7847e-02, 9.8828e-01, 1.4941e-01, 1.7765e-02],
  #             [-7.6074e-01, -1.1792e-01, 6.3818e-01, -6.8617e-02],
  #             [6.4844e-01, -9.5825e-02, 7.5537e-01, 9.7462e-01],
  #             [0.0000e00, 0.0000e00, 0.0000e00, 1.0000e00],
  #         ],
  #     ]],
  #     device=device,
  # )
  # Nk = len(Ks)
  # backgrounds = torch.zeros((1, Nk, 3), device=device, dtype=torch.float32)
  # width = 128
  # height = 128
  # pudb.set_trace()
  # renders = rasterization_2dgs(
  #     means=means,
  #     quats=quats,
  #     scales=scales,
  #     opacities=opacities,
  #     colors=colors,
  #     viewmats=viewmats,
  #     Ks=Ks,
  #     width=width,
  #     height=height,
  #     backgrounds=backgrounds,
  # )[:3]
  # print(renders)

  means = torch.randn((100, 3), device=device)
  quats = torch.randn((100, 4), device=device)
  scales = torch.rand((100, 3), device=device) * 0.1
  colors = torch.rand((100, 3), device=device)
  opacities = torch.rand((100,), device=device)
  # define cameras
  viewmats = torch.eye(4, device=device)[None, :, :]
  Ks = torch.tensor(
      [[300.0, 0.0, 150.0], [0.0, 300.0, 100.0], [0.0, 0.0, 1.0]], device=device
  )[None, :, :]
  width, height = 300, 200
  # render
  pudb.set_trace()
  output = rasterization(
      means,
      quats,
      scales,
      opacities,
      colors,
      viewmats,
      Ks,
      width,
      height,
      packed=False,
  )
  output_2dgs = rasterization_2dgs(
      means, quats, scales, opacities, colors, viewmats, Ks, width, height
  )
  print(output)
  print(output_2dgs)


if __name__ == "__main__":
  main()
