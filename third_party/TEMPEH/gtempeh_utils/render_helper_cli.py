# conda activate gtempeh; CC=gcc-11 CXX=g++-11 CUDA_VISIBLE_DEVICES=0, python vr/perception/multiview_mesh_prediction/gtempeh_standalone/src/utils/render_helper_cli.py
from collections.abc import Sequence
import os

import einops
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import pudb
from src import data
from src.options import Options
from src.utils import file_helper
from src.utils import render_helper
import torch


@torch.no_grad()
def main(argv: Sequence[str]) -> None:

  torch.manual_seed(0)
  opt = Options()
  root = "/usr/local/google/home/mprinzler/projects/gtempeh/data/raw/2025-06-23_15-10_tbolkart_200k_tempeh_holobooth_withaccessoires_cleancams_hairproxy"
  dataset = data.HoloboothParquetDataset(
      data_source=data.HoloboothParquetChunkDataSource(
          file_dir=root,
          opt=opt,
      ),
      opt=opt,
      training=False,
  )

  # for i, sample in enumerate(ds):
  #   print(i)
  #   for k, v in sample.items():
  #     if isinstance(v, torch.Tensor):
  #       print(k, v.shape)
  #   print("\n")
  # sample = next(ds)
  # pudb.set_trace()
  # print()

  # sample = next(ds)
  # print()

  faces = np.load(
      file_helper.get_resource_filename(
          "google3/vr/perception/multiview_mesh_prediction/gtempeh/assets/gnome/triangles.npy"
      )
  )

  sample = dataset[0]
  batch = torch.utils.data.default_collate([sample])

  b, v, c, h, w = batch["image"].shape
  for ib in range(b):
    vertices = einops.repeat(
        batch["verts"][ib].cpu().numpy(), "N C -> V N C", V=v
    )
    fxfycxcy = batch["fxfycxcy"].cpu().numpy()[ib]
    camera_intrinsics = np.zeros((v, 2, 3), dtype=np.float32)
    camera_intrinsics[:, 0, 0] = fxfycxcy[:, 0] * (w - 1)
    camera_intrinsics[:, 1, 1] = fxfycxcy[:, 1] * (h - 1)
    camera_intrinsics[:, 0, 2] = fxfycxcy[:, 2] * (w - 1)
    camera_intrinsics[:, 1, 2] = fxfycxcy[:, 3] * (h - 1)
    camera_extrinsics = np.linalg.inv(batch["C2W"][ib].cpu().numpy())[:, :3, :4]

    rendering = render_helper.render_holobooth_mesh(
        vertices=vertices,
        faces=faces,
        camera_extrinsics=camera_extrinsics,
        camera_intrinsics=camera_intrinsics,
        image_size=(h, w),
        multisample_antialiasing=1,
        background_color=np.array([1.0, 1.0, 1.0]),
        enable_cull_face=False,
    )
    for iv in range(v):
      plt.imshow(rendering[iv])
      plt.savefig(f"/tmp/render_{ib}_{iv}.png")


if __name__ == "__main__":
  main([])
