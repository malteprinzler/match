# blaze run //vr/perception/multiview_mesh_prediction/gtempeh/src/utils:uv_util_cli --define cuda_target_sm60=0 --config=cuda
import matplotlib.pyplot as plt

plt.switch_backend("TkAgg")
fig = plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.close(fig)
from collections.abc import Sequence
import copy
from itertools import product
from typing import cast
from absl import app
import einops
import tqdm
import numpy as np
import pudb
import torch
from google3.pyglib import resources
from google3.pyglib.contrib.gpathlib import gpath
from google3.vr.perception.multiview_mesh_prediction.gtempeh.src.data import holobooth_dataset, serialized_holobooth_dataset
from google3.vr.perception.multiview_mesh_prediction.gtempeh.src.options import Options
from google3.vr.perception.multiview_mesh_prediction.gtempeh.src.utils import uv_util
import os


def main(argv: Sequence[str]) -> None:
  faces = np.load(
      resources.GetResourceFilename(
          "google3/vr/perception/multiview_mesh_prediction/gtempeh/assets/gnome/triangles.npy"
      )
  )
  face_uv_coords = np.load(
      resources.GetResourceFilename(
          "google3/vr/perception/multiview_mesh_prediction/gtempeh/assets/gnome/single_atlas_triangle_uvs.npy"
      )
  )

  nviews = 13
  ninputviews = 13
  nneighbors_tempeh = 50
  datauv_is_tempehuv = True
  vis = False
  opt = Options(
      num_views=nviews,
      num_input_views=ninputviews,
      input_res=128,
  )

  # # explicit dataset
  # ds = holobooth_dataset.HoloboothParquetDataset(
  #     opt=opt,
  #     data_source=holobooth_dataset.HoloboothParquetChunkDataSource(
  #         file_dir="/usr/local/google/home/mprinzler/projects/gtempeh/data/raw/2025-05-13_jbednarik_synth1_tempeh_holobooth_withaccessoires_cleancams_withbg",
  #         stage1verts_dir="/usr/local/google/home/mprinzler/projects/gtempeh/data/mesh_predictions/tempeh_mixedsynth200kreal157k_onlygeom_global_13views_128p/train/predictions_split/00973000/05_21-gtempeh_200ksubj_withaccessories_gtuv_128p_escapeall_sebastian",
  #         opt=opt,
  #     ),
  #     training=False,
  # )
  # dataloader = torch.utils.data.DataLoader(
  #     ds,
  #     batch_size=1,
  #     shuffle=False,
  #     num_workers=0,
  #     pin_memory=True,
  # )

  # serialized dataset
  dataloader = serialized_holobooth_dataset.SerializedHoloboothDataloader(
      data_proto_path=gpath.GPath(
          "/usr/local/google/home/mprinzler/projects/gtempeh/data/serialized/train/06_25-gtempeh_newreal_157ksubj_tempehhairproxyuv_512p_withmasks.sstable-00000-of-10000"
      ),
      batch_size=1,
      is_training=False,
      shuffle=False,
      shuffle_buffer_size=-1,
      nviews=opt.num_views,
      ninputviews=opt.num_input_views,
      drop_remainder=False,
  )
  sample = next(dataloader)

  # coverage_scores = list()
  # for sample in tqdm.tqdm(dataloader):
  #   sample_tempehuv = copy.deepcopy(sample)
  #   if not datauv_is_tempehuv:
  #     uv_renders_tempehuv = uv_util.render_uvmaps_from_sample(
  #         sample, faces, face_uv_coords, use_gtverts=False
  #     )
  #     sample_tempehuv["uv"] = einops.rearrange(
  #         torch.from_numpy(uv_renders_tempehuv), "b v h w c -> b v c h w"
  #     )

  #   uv_renders_gt = uv_util.render_uvmaps_from_sample(
  #       sample, faces, face_uv_coords, use_gtverts=True
  #   )
  #   sample_gtuv = copy.deepcopy(sample)
  #   sample_gtuv["uv"] = einops.rearrange(
  #       torch.from_numpy(uv_renders_gt), "b v h w c -> b v c h w"
  #   )

  #   uv_match_results = dict()
  #   for kind, sample in [("gt", sample_gtuv), ("tempeh", sample_tempehuv)]:
  #     uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores = (
  #         uv_util.get_img_uv_patch_correspondences(
  #             uv_renders=sample["uv"],
  #             uv_res=opt.uv_res,
  #             patch_size=opt.patch_size,
  #             n_neighbors=nneighbors_tempeh
  #             if kind == "tempeh"
  #             else opt.n_neighbors,
  #             downsample_for_computation=opt.input_res // 128,
  #             return_uv_patch_mask=True,
  #         )
  #     )
  #     uv_match_results[kind] = dict(
  #         uv_patch_masks=uv_patch_masks,
  #         uv_img_patch_match_idcs=uv_img_patch_match_idcs,
  #         uv_img_patch_match_scores=uv_img_patch_match_scores,
  #     )

  #   # calculate iou of img patch idcs
  #   b, nv_uv, nh_uv = uv_match_results["gt"]["uv_img_patch_match_idcs"].shape[
  #       :3
  #   ]
  #   batch_coverage_scores = list()
  #   for i_b, i_nv_uv, i_nh_uv in product(
  #       range(b),
  #       range(nv_uv),
  #       range(nh_uv),
  #   ):
  #     gt_patch_ids = set(
  #         uv_match_results["gt"]["uv_img_patch_match_idcs"][
  #             i_b, i_nv_uv, i_nh_uv
  #         ][
  #             uv_match_results["gt"]["uv_img_patch_match_scores"][
  #                 i_b, i_nv_uv, i_nh_uv
  #             ]
  #             > 0
  #         ].tolist()
  #     )
  #     tempehuv_patch_ids = set(
  #         uv_match_results["tempeh"]["uv_img_patch_match_idcs"][
  #             i_b, i_nv_uv, i_nh_uv
  #         ][
  #             uv_match_results["tempeh"]["uv_img_patch_match_scores"][
  #                 i_b, i_nv_uv, i_nh_uv
  #             ]
  #             > 0
  #         ].tolist()
  #     )
  #     if len(gt_patch_ids) > 0:
  #       coverage_score = len(
  #           gt_patch_ids.intersection(tempehuv_patch_ids)
  #       ) / len(gt_patch_ids)
  #       batch_coverage_scores.append(coverage_score)

  #       if vis and i_nv_uv == 26:
  #         # visualize tempeh uv img patch matches and gt uv img patch matches
  #         uv_pixel_size = 1 / (opt.uv_res - 1)
  #         uv_sqrt_n_patches = opt.uv_res // opt.patch_size
  #         uv_patch_edges_u = (
  #             -0.5 * uv_pixel_size
  #             + uv_pixel_size
  #             * opt.patch_size
  #             * np.arange(uv_sqrt_n_patches + 1)
  #         )  # (N_patches + 1)
  #         uv_patch_edges_v = (
  #             1 - uv_patch_edges_u
  #         )  # uv origin is at bottom left (sqrt_n_patches + 1)
  #         h, w = sample_gtuv["uv"].shape[-2:]

  #         uv_scatters = einops.rearrange(face_uv_coords, "F V C -> (F V) C")

  #         nrows = 3
  #         ncols = opt.num_input_views + 1
  #         s = 5
  #         fig, ax = plt.subplots(
  #             nrows=nrows,
  #             ncols=ncols,
  #             figsize=(s * ncols, s * nrows),
  #             squeeze=False,
  #         )
  #         ax[0, 0].scatter(uv_scatters[:, 0], uv_scatters[:, 1], s=0.2)
  #         ax[0, 0].add_patch(
  #             plt.Rectangle(
  #                 (uv_patch_edges_u[i_nh_uv], uv_patch_edges_v[i_nv_uv]),
  #                 uv_pixel_size * opt.patch_size,
  #                 uv_pixel_size * opt.patch_size,
  #                 linewidth=1,
  #                 edgecolor="r",
  #                 facecolor="none",
  #                 fill=False,
  #             )
  #         )

  #         # drawing uv maps
  #         for i_v in range(opt.num_input_views):
  #           ax[0, i_v + 1].imshow(sample["image"][i_b, i_v].permute(1, 2, 0))

  #           for row_idx, sample, kind in zip(
  #               range(1, 3), [sample_gtuv, sample_tempehuv], ["gt", "tempeh"]
  #           ):
  #             col_idx = i_v + 1
  #             ax[row_idx, col_idx].imshow(
  #                 np.concatenate(
  #                     (
  #                         einops.rearrange(
  #                             sample["uv"][i_b, i_v], "C H W -> H W C"
  #                         )[..., :2]
  #                         .cpu()
  #                         .numpy(),
  #                         uv_match_results[kind]["uv_patch_masks"][
  #                             i_b, i_v, i_nv_uv, i_nh_uv
  #                         ]
  #                         .cpu()
  #                         .numpy()[..., None],
  #                     ),
  #                     axis=-1,
  #                 )
  #             )

  #         # drawing uv patch matches
  #         gt_imgpatch_ids = uv_match_results["gt"]["uv_img_patch_match_idcs"][
  #             i_b, i_nv_uv, i_nh_uv
  #         ][
  #             uv_match_results["gt"]["uv_img_patch_match_scores"][
  #                 i_b, i_nv_uv, i_nh_uv
  #             ]
  #             > 0
  #         ].tolist()
  #         tempeh_imgpatch_ids = uv_match_results["tempeh"][
  #             "uv_img_patch_match_idcs"
  #         ][i_b, i_nv_uv, i_nh_uv][
  #             uv_match_results["tempeh"]["uv_img_patch_match_scores"][
  #                 i_b, i_nv_uv, i_nh_uv
  #             ]
  #             > 0
  #         ].tolist()
  #         gt_patch_colors = [
  #             "g" if id in tempeh_imgpatch_ids else "r"
  #             for id in gt_imgpatch_ids
  #         ]
  #         tempeh_patch_colors = [
  #             "g" if id in gt_imgpatch_ids else "r"
  #             for id in tempeh_imgpatch_ids
  #         ]
  #         for row_idx, patches, patch_colors in zip(
  #             range(1, 3),
  #             [gt_imgpatch_ids, tempeh_imgpatch_ids],
  #             [gt_patch_colors, tempeh_patch_colors],
  #         ):

  #           for p, c in zip(patches, patch_colors):
  #             vidx, patch_row_idx, patch_col_idx = uv_util.unflatten_patch_idcs(
  #                 p, h, w, opt.patch_size
  #             )
  #             patch_tl = (
  #                 patch_col_idx * opt.patch_size,
  #                 patch_row_idx * opt.patch_size,
  #             )
  #             ax[row_idx, vidx + 1].add_patch(
  #                 plt.Rectangle(
  #                     patch_tl,
  #                     opt.patch_size,
  #                     opt.patch_size,
  #                     linewidth=1,
  #                     edgecolor=c,
  #                     facecolor="none",
  #                     fill=False,
  #                 )
  #             )
  #         for a in ax.flatten():
  #           a.set_xticks([])
  #           a.set_yticks([])
  #         ax[1, 0].set_ylabel("gt")
  #         ax[2, 0].set_ylabel("tempeh")
  #         fig.suptitle(f"coverage score: {coverage_score:.3f}")
  #         plt.tight_layout()
  #         plt.show()
  #         plt.close(fig)
  #         # outpath = f"/tmp/vis/uv_patch_match_scores_{i:03d}_{j:03d}.png"
  #         # plt.savefig(outpath)
  #         # print(outpath)
  #         # plt.close(fig)

  #   coverage_scores.append(np.mean(batch_coverage_scores))
  # print(f"\nAverage coverage score: {np.mean(coverage_scores):.3f}")
  # pudb.set_trace()
  # print()

  # ###
  # # find proper nr of neighbors
  # ###
  # # yielded ~150
  # uv_res = 256
  # patch_size = 8  # TODO
  # n_neighbors = 200
  # input_res = 256

  # uv_renders = uv_util.render_uvmaps_from_sample(sample, faces, face_uv_coords)
  # uv_renders = einops.rearrange(
  #     torch.from_numpy(uv_renders), "b v h w c -> b v c h w"
  # )
  # b, v, _, h, w = uv_renders.shape

  # uv_renders = einops.rearrange(
  #     torch.nn.functional.interpolate(
  #         einops.rearrange(uv_renders, "b v c h w -> (b v) c h w"),
  #         scale_factor=input_res / h,
  #         mode="nearest",
  #     ),
  #     "(b v) c h w -> b v c h w",
  #     b=b,
  #     v=v,
  # )

  # uv_renders = uv_renders[:, :ninputviews]  # (B, V, 3, H, W)
  # downsample_for_computation = input_res // 128
  # uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores = (
  #     uv_util.get_img_uv_patch_correspondences(
  #         uv_renders,
  #         uv_res,
  #         patch_size,
  #         n_neighbors=n_neighbors,
  #         downsample_for_computation=downsample_for_computation,
  #         return_uv_patch_mask=True,
  #     )
  # )
  # uv_img_patch_match_nozeroscores = (uv_img_patch_match_scores > 0).int()
  # plt.hist(
  #     einops.rearrange(
  #         uv_img_patch_match_nozeroscores,
  #         "b nv_uv nh_uv n_neighbors -> (b nv_uv nh_uv) n_neighbors",
  #     ).sum(dim=-1),
  #     bins=100,
  # )
  # outpath = "/tmp/vis/uv_patch_match_nozeroscores.png"
  # plt.savefig(outpath)
  # print(outpath)
  # plt.close()

  ###
  # visualization
  ###

  uv_res = 512
  patch_size = 8  # TODO
  n_neighbors = 10
  img_res = 512

  sample = holobooth_dataset.resize_data_chunk(sample, img_res)
  uv_renders = uv_util.render_uvmaps_from_sample(sample, faces, face_uv_coords)
  uv_renders = einops.rearrange(
      torch.from_numpy(uv_renders), "b v h w c -> b v c h w"
  )

  uv_renders = uv_renders[:, :ninputviews]  # (B, V, 3, H, W)
  h = uv_renders.shape[-2]
  downsample_for_computation = h // 256
  uv_patch_masks, uv_img_patch_match_idcs, uv_img_patch_match_scores = (
      uv_util.get_img_uv_patch_correspondences(
          uv_renders,
          uv_res,
          patch_size,
          n_neighbors=n_neighbors,
          downsample_for_computation=downsample_for_computation,
          return_uv_patch_mask=True,
      )
  )  # (B, V, nv_uv, nh_uv, H, W), (B, nv_uv, nh_uv, n_neighbors), (B, nv_uv, nh_uv, n_neighbors)

  # visualize uv_patch_match_scores
  uv_pixel_size = 1 / (uv_res - 1)
  uv_sqrt_n_patches = uv_res // patch_size
  uv_patch_edges_u = (
      -0.5 * uv_pixel_size
      + uv_pixel_size * patch_size * np.arange(uv_sqrt_n_patches + 1)
  )  # (N_patches + 1)
  uv_patch_edges_v = (
      1 - uv_patch_edges_u
  )  # uv origin is at bottom left (sqrt_n_patches + 1)
  h, w = uv_renders.shape[-2:]

  uv_scatters = einops.rearrange(face_uv_coords, "F V C -> (F V) C")
  n_imgs = ninputviews + 1
  plot_grid_size = int(np.ceil(np.sqrt(n_imgs)))
  for i, j in product(
      range(uv_sqrt_n_patches // 2, uv_sqrt_n_patches),
      range(uv_sqrt_n_patches // 2, uv_sqrt_n_patches),
  ):
    fig, ax = plt.subplots(
        ncols=plot_grid_size,
        nrows=plot_grid_size,
        figsize=(20, 20),
        squeeze=False,
    )
    ax = ax.flatten()
    ax[0].scatter(uv_scatters[:, 0], uv_scatters[:, 1], s=0.2)
    ax[0].add_patch(
        plt.Rectangle(
            (uv_patch_edges_u[j], uv_patch_edges_v[i]),
            uv_pixel_size * patch_size,
            uv_pixel_size * patch_size,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            fill=False,
        )
    )
    for k in range(ninputviews):
      ax[k + 1].imshow(
          np.concatenate(
              (
                  einops.rearrange(uv_renders[0, k], "C H W -> H W C")[..., :2],
                  uv_patch_masks[
                      0,
                      k,
                      i,
                      j,
                  ]
                  .cpu()
                  .numpy()[..., None],
              ),
              axis=-1,
          )
      )

    for k in range(n_neighbors):
      neighbor_idx = uv_img_patch_match_idcs[0, i, j, k]
      neighbor_score = uv_img_patch_match_scores[0, i, j, k]
      vidx, row_idx, col_idx = uv_util.unflatten_patch_idcs(
          neighbor_idx, h, w, patch_size
      )
      patch_tl = (col_idx * patch_size, row_idx * patch_size)
      ax[vidx + 1].add_patch(
          plt.Rectangle(
              patch_tl,
              patch_size,
              patch_size,
              linewidth=1,
              edgecolor="r",
              facecolor="none",
              fill=False,
          )
      )
      ax[vidx + 1].text(
          patch_tl[0].cpu().item() + 0.5 * patch_size,
          patch_tl[1].cpu().item(),
          f"{k}, {neighbor_score:.3f}",
          fontsize=5,
          ha="center",
          color="white",
      )
    plt.tight_layout()
    outpath = f"/tmp/vis/uv_patch_match_scores_{i:03d}_{j:03d}.png"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    print(outpath)
    plt.close(fig)


if __name__ == "__main__":
  app.run(main)
