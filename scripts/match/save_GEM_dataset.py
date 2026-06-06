import os
os.environ['PYOPENGL_PLATFORM'] = 'egl' 
from torch._tensor import Tensor
from typing import Any


import argparse 
from collections import defaultdict
from collections.abc import Sequence
import os
from absl import app
from absl import flags
from absl import logging as absl_logging
import accelerate
import einops
import gin
import mediapy as media
import numpy as np
import pudb
import tensorflow as tf
import torch
import tqdm
from match.utils import data_util, file_util, general_util
from match.options import Options
from match import models, data
from match.utils import general_util, render_util, file_util, gin_util, data_util, geo_util
import copy
from pyvirtualdisplay import Display
from PIL import Image
from match.runner import MatchRunner
import json
import math
from copy import deepcopy

def split_transform(transform: dict, train_ratio=0.7):
    db = transform
    
    # init db for each division
    db_train = {k: v for k, v in db.items() if k not in ['frames', 'timestep_indices', 'camera_indices']}
    db_train['frames'] = []
    db_val = deepcopy(db_train)
    db_test = deepcopy(db_train)

    # divide timesteps
    nt = len(db['timestep_indices'])
    assert 0 < train_ratio <= 1
    nt_train = int(np.ceil(nt * train_ratio))
    nt_test = nt - nt_train

    # record number of timesteps
    timestep_indices = sorted(db['timestep_indices'])
    db_train['timestep_indices'] = timestep_indices[:nt_train]
    db_val['timestep_indices'] = timestep_indices[:nt_train]  # validation set share the same timesteps with training set
    db_test['timestep_indices'] = timestep_indices[nt_train:]

    if len(db['camera_indices']) > 1:
        # when having multiple cameras, leave one camera for validation (novel-view sythesis)

        # use the first camera for validation
        db_train['camera_indices'] = db['camera_indices'][1:]
        db_val['camera_indices'] = db['camera_indices'][:1]
        db_test['camera_indices'] = db['camera_indices']
    else:
        # when only having one camera, train and validation set share the same camera
        db_train['camera_indices'] = db['camera_indices']
        db_val['camera_indices'] = db['camera_indices']
        db_test['camera_indices'] = db['camera_indices']

    # fill data by timestep index
    range_train = range(db_train['timestep_indices'][0], db_train['timestep_indices'][-1]+1) if nt_train > 0 else []
    range_test = range(db_test['timestep_indices'][0], db_test['timestep_indices'][-1]+1) if nt_test > 0 else []
    for f in db['frames']:
        if f['timestep_index'] in range_train:
            if f['camera_index'] in db_train['camera_indices']:
                db_train['frames'].append(f)
            if f['camera_index'] in db_val['camera_indices']:
                db_val['frames'].append(f)
            if not (f['camera_index'] in db_train['camera_indices'] or f['camera_index'] in db_val['camera_indices']):
                raise ValueError(f"Unknown camera index: {f['camera_index']}")
        elif f['timestep_index'] in range_test:
            db_test['frames'].append(f)
            assert f['camera_index'] in db_test['camera_indices'], f"Unknown camera index: {f['camera_index']}"
        else:
            raise ValueError(f"Unknown timestep index: {f['timestep_index']}")
    
    return dict[str, dict](train=db_train, val=db_val, test=db_test)


def nerf_camparams_from_match_camparams(cam_params, H:int, W:int, extrinsic_type='w2c'):
    '''
    Coordinate systems:
        World nerf: x right, y up, z face looking direction
        World match: x right, y down, z inv face looking direction
        Cam nerf: x right, y up, z inverse view direction
        Cam match: x right, y: down, z: view direction 


    Args:
        cam_params: 
            'C2W': torch.Tensor(...,4,4)
            'fxfycxcy': torch.Tensor(..., 4)
        H
        W
    
    Returns:
        nerf cam params
            'intrinsic': torch.Tensor(..., 3,3)
            'extrinsic': torch.Tensor(..., 3,4) 
    
    '''
    c2w = cam_params['C2W']
    device = c2w.device
    dtype=c2w.dtype
    w2c = geo_util.invert_c2w(c2w)
    invert_yz = torch.diag(torch.tensor([1, -1, -1, 1], device=device, dtype=dtype))
    w2c = einops.einsum(w2c, invert_yz, 'v i j, j k -> v i k')  # invert y and z axis in world space
    w2c = einops.einsum(invert_yz, w2c, 'i j, v j k -> v i k')  # invert y and z axis in camera space
    c2w = geo_util.invert_c2w(w2c)
    w2c = w2c[..., :3, :]
    c2w = c2w[..., :3, :]
    if extrinsic_type == "w2c":
        extrinsic = w2c
    elif extrinsic_type == "c2w":
        extrinsic = c2w
    else:
        raise NotImplementedError(f"Unknown extrinsic type: {extrinsic_type}")

    V = len(w2c)
    intrinsics = torch.repeat_interleave(torch.eye(3, device=c2w.device, dtype=c2w.dtype)[None], V, dim=0)
    fx, fy, cx, cy = torch.unbind(cam_params['fxfycxcy'], dim=-1)
    fx = fx * W
    fy = fy * H
    cx = cx * W
    cy = cy * H
    cy = H - cy # flipping y axis in screen space
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fx
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    return dict[str, Tensor](intrinsic=intrinsics, extrinsic=extrinsic)

@gin.configurable()
def save_GEM_dataset(runner: MatchRunner, out_dir: str = None, store_config_files: list[str] = None, skip_predictions: bool = False) -> None:
  """saving data for distillation dataset

  will generate folder structure like:
  root/
    ...
    {subject_id_0}/
      {sequence_id_0}/
        images/
          {frame_id}_00.png
          {frame_id}_01.png
          ...
        fg_masks/
          {frame_id}_00.png
          {frame_id}_01.png
          ...
        cameras/
          {frame_id}.json
        gaussians/
          {frame_id}.pt  # dict with keys:
                         #   - xyz: (1, 3, H, W)
                         #   - rgb: (1, 3, H, W)
                         #   - scale: (1, 1, H, W)
                         #   - rotation: (1, 4, H, W)
                         #   - opacity: (1, 1, H, W)
                         #   - uv: (1, 3, H, W)
                         #   - mask: (1, 1, H, W)  ... mask which texels are used as gsplats
        visualizations/
          {frame_id}.jpg
        transforms.json
        transforms_train.json
        transforms_val.json
        transforms_test.json
      {sequence_id_1}/
        ...
    {subject_id_1}/
      ...
  
  """
  opt = runner.opt
  import pudb; pudb.set_trace()

  if out_dir is None:
    out_dir = runner.exp_dir / "distillation_dataset" / f"iter_{runner.pretrained_model_iter:08d}"
  out_dir = file_util.Path(out_dir)
  runner.logger.info(f"Saving distillation dataset to [{out_dir}]\n")
  
  out_dir.mkdir(parents=True, exist_ok=True)
  # saving configs
  for i in range(len(store_config_files)):
    try:
      file_util.copy(store_config_files[i], out_dir / f"config_{i}.gin", True)
    except Exception as e:
      absl_logging.info(
          f"Failed to copy gin config from [{store_config_files[i]}] to"
          f" [{out_dir / f'config_{i}.gin'}]\n"
      )
      pass

  # split val datasets across processes
  process_idx = int(os.environ.get('CONDOR_Process', '0'))
  world_size = int(os.environ.get('CONDOR_WORLD_SIZE', '1'))
  runner.split_val_sets_across_processes(process_idx=process_idx, world_size=world_size)
  runner.init_val_loaders()
  runner.accelerate_prepare()
  runner.eval()

  if not skip_predictions:
    compiled_model = runner.get_compiled_model()
    unwrapped_model = runner.get_unwrapped_model()
  else:
    compiled_model = None
    unwrapped_model = None

  processed_subjects_sequences = set()

  file_util.makedirs(out_dir, exist_ok=True)
  with torch.autocast('cuda', dtype=torch.float16):
    with torch.no_grad():
      for dataset_name, loader in zip(runner.dataset_names_val, runner.val_loaders):
        runner.logger.info(f"Processing dataset [{dataset_name}]...")
        for j, batch in enumerate(tqdm.tqdm(loader, desc=f"Dataset [{dataset_name}]")):
          batch = data_util.MatchBatch(batch)
          input_batch = batch
          output_batch = batch.cat_extra_keys()
          
          b, v, _, h, w = output_batch["image"].shape
          out_path_images = [[out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'images' / f"{output_batch['frame'][i_s]}_{i_v:02d}.png" for i_v in range(v)]  for i_s in range(len(output_batch['subject']))]
          out_path_masks = [[out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'fg_masks' / f"{output_batch['frame'][i_s]}_{i_v:02d}.png" for i_v in range(v)]  for i_s in range(len(output_batch['subject']))]
          out_path_segs = [[out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'sapiens_seg' / f"{output_batch['frame'][i_s]}_{i_v:02d}.png" for i_v in range(v)]  for i_s in range(len(output_batch['subject']))]
          out_path_cameras = [out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'cameras' / f"{output_batch['frame'][i_s]}.json" for i_s in range(len(output_batch['subject']))]
          out_path_gaussians = [out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'gaussians' / f"{output_batch['frame'][i_s]}.pt" for i_s in range(len(output_batch['subject']))]
          out_path_viss = [out_dir / output_batch['subject'][i_s] / output_batch['sequence'][i_s] / 'visualizations' / f"{output_batch['frame'][i_s]}.jpg" for i_s in range(len(output_batch['subject']))]

          for i_s in range(b):
            # keeping track of which subjects where processed for later 
            processed_subjects_sequences.add(output_batch['subject'][i_s]+'/' + output_batch['sequence'][i_s])

          # skip batches where all outputs already exist
          required_paths = (
              out_path_cameras
              + [p for sublist in out_path_images for p in sublist]
              + [p for sublist in out_path_masks for p in sublist]
              + [p for sublist in out_path_segs for p in sublist]
          )
          if not skip_predictions:
            required_paths = (
                out_path_gaussians + out_path_viss + required_paths
            )
          if all(p.exists() for p in required_paths):
            continue

          input_batch = general_util.batch_to_device(input_batch, runner.device)
          if not skip_predictions:
            gaussian_parameters = compiled_model(input_batch, runner.weight_dtype, func_name='forward_gaussians')
            masked_gaussian_parameters = unwrapped_model.mask_gaussians(gaussian_parameters)
            verts_pred = unwrapped_model.gaussians2mesh(gaussian_parameters)

            ########
            # create visualization
            ########
            vis_idcs = (
                torch.tensor(np.linspace(0, 12, 5)[1:-1])
                .round()
                .to(dtype=torch.long, device=runner.accelerator.device)
            )
            v_vis = len(vis_idcs)
            vis_c2w = torch.index_select(input_batch["C2W"], dim=1, index=vis_idcs)
            vis_fxfycxcy = torch.index_select(
                input_batch["fxfycxcy"], dim=1, index=vis_idcs
            )
            render_img = unwrapped_model.gs_renderer.render(
                masked_gaussian_parameters,
                vis_c2w,
                vis_fxfycxcy,
                height=h,
                width=w,
                bg_color=(1.0, 1.0, 1.0),
            )["image"]
            render_gauss = unwrapped_model.gs_renderer.render(
                masked_gaussian_parameters,
                vis_c2w,
                vis_fxfycxcy,
                height=h,
                width=w,
                render_gauss=True,
            )["image"]

            extrinsics = np.linalg.inv(vis_c2w.cpu().numpy())[:, :, :3, :4]
            extrinsics = einops.rearrange(extrinsics, "B V H W -> (B V) H W")
            meshvis_fxfycxcy = vis_fxfycxcy.cpu().numpy()
            meshvis_fxfycxcy = einops.rearrange(meshvis_fxfycxcy, "B V C -> (B V) C")

            intrinsics = np.zeros((b * v_vis, 2, 3))
            intrinsics[:, 0, 0] = meshvis_fxfycxcy[:, 0]
            intrinsics[:, 0, 2] = meshvis_fxfycxcy[:, 2]
            intrinsics[:, 1, 1] = meshvis_fxfycxcy[:, 1]
            intrinsics[:, 1, 2] = meshvis_fxfycxcy[:, 3]
            intrinsics[:, 0] *= w - 1
            intrinsics[:, 1] *= h - 1
            distortions = np.zeros((b * v_vis, 5), dtype=np.float32)
            faces = unwrapped_model.template_triangles.cpu().numpy().astype(np.int32)
            render_mesh = render_util.render_mesh(
                vertices=einops.repeat(verts_pred, "B N C -> (B V) N C", V=v_vis)
                .cpu()
                .numpy()
                .astype(np.float32),
                faces=faces,
                camera_extrinsics=extrinsics.astype(np.float32),
                camera_intrinsics=intrinsics.astype(np.float32),
                camera_distortions=distortions.astype(np.float32),
                image_size=(h, w),
                multisample_antialiasing=1,
                background_color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                enable_cull_face=False,
            )
            render_mesh = einops.rearrange(
                torch.from_numpy(render_mesh).to(render_img),
                "(B V) H W C -> B V C H W",
                B=b,
                V=v_vis,
            )  # (B H W C)

            # rearranging and resizing
            vis_input = torch.index_select(input_batch["image"], dim=1, index=vis_idcs).to(
                render_img
            )
            if opt.rot90:
              vis_input = torch.rot90(vis_input, dims=(-2, -1))
              render_img = torch.rot90(render_img, dims=(-2, -1))
              render_gauss = torch.rot90(render_gauss, dims=(-2, -1))
              render_mesh = torch.rot90(render_mesh, dims=(-2, -1))

            vis_input = einops.rearrange(vis_input, "B V C H W -> B C H (V W)")
            render_img = einops.rearrange(render_img, "B V C H W -> B C H (V W)")
            render_gauss = einops.rearrange(render_gauss, "B V C H W -> B C H (V W)")
            render_mesh = einops.rearrange(render_mesh, "B V C H W -> B C H (V W)")

            vis = torch.cat([vis_input, render_img, render_gauss, render_mesh], dim=-2)
            w_vis = 384
            h_vis = int(w_vis * vis.shape[-2]/vis.shape[-1])
            vis = torch.nn.functional.interpolate(
                vis,
                size=(h_vis, w_vis),
                mode="bilinear",
                align_corners=False,
            )

          ##########################

          # saving outputs
          for i in range(b):
            out_path_cam_i = out_path_cameras[i]
            out_path_cam_i_tmp = out_path_cam_i.parent / (
                "TMP_" + out_path_cam_i.name
            )

            cameras = dict([
                (k, output_batch[k][i].tolist()) for k in ["C2W", "fxfycxcy"]
            ])

            # save gaussians and visualization only when model predictions are run
            if not skip_predictions:
              out_path_pred_i = out_path_gaussians[i]
              out_path_vis_i = out_path_viss[i]
              out_path_pred_i_tmp = out_path_pred_i.parent / (
                  "TMP_" + out_path_pred_i.name
              )
              out_path_vis_i_tmp = out_path_vis_i.parent / (
                  "TMP_" + out_path_vis_i.name
              )

              sample_gaussian_parameters = dict(
                  [(k, gaussian_parameters[k][i]) for k in gaussian_parameters.keys()]
              )
              sample_gaussian_parameters["mask"] = einops.rearrange(
                  unwrapped_model.uv_grid_mask, "H W C -> 1 C H W"
              )

              out_path_pred_i.parent.mkdir(parents=True, exist_ok=True)
              with file_util.open_file(out_path_pred_i_tmp, "wb") as f:
                torch.save(sample_gaussian_parameters, f)
              file_util.rename(out_path_pred_i_tmp, out_path_pred_i, overwrite=True)

              out_path_vis_i.parent.mkdir(exist_ok=True, parents=True)
              media.write_image(
                  out_path_vis_i_tmp,
                  einops.rearrange(vis[i], "C H W -> H W C").cpu().numpy(),
              )
              file_util.rename(out_path_vis_i_tmp, out_path_vis_i, overwrite=True)

            out_path_cam_i.parent.mkdir(parents=True, exist_ok=True)
            with file_util.open_file(out_path_cam_i_tmp, "w") as f:
              json.dump(cameras, f, indent='\t')
            file_util.rename(out_path_cam_i_tmp, out_path_cam_i, overwrite=True)

            for iv in range(v):
              for k, outpath in [('image', out_path_images[i][iv]), 
                                ('mask', out_path_masks[i][iv]), 
                                ('sg_parts', out_path_segs[i][iv]),
                                ]:
                outpath.parent.mkdir(exist_ok=True, parents=True)
                outpath_tmp = outpath.parent / (
                    "TMP_" + outpath.name
                )
                
                img = output_batch[k][i][iv]
                
                if k == 'image': # making image background white
                  img = img * output_batch['mask'][i][iv] + (1-output_batch['mask'][i][iv])

                img = img.cpu().permute(1,2,0).numpy()
                if k == 'sg_parts': # segmentations are already in uint8 format 
                  img = img[..., 0]
                else: 
                  img = img * 255
                  img = np.squeeze(np.clip(np.round(img), 0, 255).astype(np.uint8))
                Image.fromarray(img).save(outpath_tmp)
                file_util.rename(outpath_tmp, outpath, overwrite=True)

  # making the transforms*.json files
  for subj_seq in sorted(processed_subjects_sequences):
    subj_seq_root = out_dir / subj_seq
    img_dir = subj_seq_root / 'images'
    cam_dir = subj_seq_root / 'cameras'
    img_paths = [p for p in sorted(img_dir.iterdir()) if p.is_file() and not 'tmp' in p.name.lower()]
    frames = list(sorted(set([p.name.split('_')[0] for p in img_paths])))

    transforms_frames = list()
    for img_path in img_paths:
      timestep_id = img_path.name.split('_')[0]
      timestep_index = frames.index(timestep_id)
      camera_id = img_path.name.split('_')[1].split('.')[0]
      camera_index = int(camera_id)
      cam_path = cam_dir / f'{timestep_id}.json'
      with cam_path.open('r') as f:
        cam_info = json.load(f)
      for k, v in cam_info.items():
        if isinstance(v, list):
          cam_info[k] = torch.tensor(v)
      img = Image.open(img_path)
      h = img.height
      w = img.width
      nerf_camparams = nerf_camparams_from_match_camparams(cam_info, h, w, extrinsic_type='c2w')
      extrinsic = nerf_camparams['extrinsic'][camera_index]
      transform_matrix = torch.cat([extrinsic, torch.tensor([[0,0,0,1]])], dim=0).numpy()
      intrinsic = nerf_camparams['intrinsic'][camera_index].double().cpu().numpy()
      cx = intrinsic[0, 2]
      cy = intrinsic[1, 2]
      fl_x = intrinsic[0, 0]
      fl_y = intrinsic[1, 1]
      angle_x = math.atan(w / (fl_x * 2)) * 2
      angle_y = math.atan(h / (fl_y * 2)) * 2

      transforms_frames.append(
        dict(
          timestep_index = timestep_index,
          timestep_id = timestep_id,
          camera_index=camera_index,
          camera_id = camera_id,
          cx=cx,
          cy=cy,
          fl_x = fl_x,
          fl_y = fl_y,
          h = h,
          w =w,
          camera_angle_x = angle_x,
          camera_angle_y = angle_y,
          transform_matrix = transform_matrix.tolist(),
          file_path = str(img_path.relative_to(subj_seq_root).as_posix()),                
        )
      ) 
    transforms = dict(
      frames=transforms_frames,
      cx = transforms_frames[0]['cx'],
      cy = transforms_frames[0]['cy'],
      fl_x = transforms_frames[0]['fl_x'],
      fl_y = transforms_frames[0]['fl_y'],
      h = transforms_frames[0]['h'],
      w = transforms_frames[0]['w'],
      camera_angle_x = transforms_frames[0]['camera_angle_x'],
      camera_angle_y = transforms_frames[0]['camera_angle_y'],
      timestep_indices = list(sorted(set([frame['timestep_index'] for frame in transforms_frames]))),
      camera_indices = list(sorted(set([frame['camera_index'] for frame in transforms_frames]))),
    )
    transform_splits = split_transform(transforms)

    with (subj_seq_root / 'transforms.json').open('w') as f:
      json.dump(transforms, f, indent='\t')

    for division, db in transform_splits.items():
      with (subj_seq_root / f'transforms_{division}.json').open('w') as f:
        json.dump(db, f, indent='\t')

  


def main(argv: list[str]|None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_configs", action="append", default=[])
    parser.add_argument("--gin_bindings", action="append", default=[])
    FLAGS = parser.parse_args(argv)

    # explicitly initialize the warm pool as the first line of main!
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=None, skip_unknown=True
    )
    opt = Options()
    runner = MatchRunner(opt)
    save_GEM_dataset(runner, store_config_files=FLAGS.gin_configs)
    runner.graceful_exit(0)



if __name__ == "__main__":
  main()
