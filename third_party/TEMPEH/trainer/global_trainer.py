"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import ast
import json
import os
import time
import random
from os.path import join
import plotly.express as px
import plotly.graph_objects as go
import pudb
import torch
import numpy as np
from torch.autograd import Variable
from utils.utils import print_memory, to_numpy, get_time_string
from utils.mesh_renderer import render_mesh, dist_to_rgb
from utils.data_augment import get_subset_views
from utils.point_to_surface_loss import PointToSurfaceLoss, compute_s2m_distance
from utils.edge_loss import EdgeLoss
from gtempeh_utils import uv_util, file_helper
import tqdm
from trainer.base_trainer import BaseTrainer
from option_handler.train_options_global import TrainOptions
import kaolin
from psbody.mesh import Mesh
from utils.mesh_sampling import MeshSampler
import einops
from datasets import data_utils
from pathlib import Path
from PIL import Image
import torchvision
from match.utils import data_util
from match.data import AvaSingleCaptureDataset, NersembleSingleCaptureDataset, NersembleMultiCaptureDataset, AvaMultiCaptureDataset, MultiDataset
from utils import normal_loss, point_to_point_loss



mean_np = np.array([0.485, 0.456, 0.406], dtype=np.float32)
std_np  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
mean = torch.from_numpy(mean_np)
std  = torch.from_numpy(std_np)

def denormalize_image(image):

    # assume image in (H,W,3) in numpy array or (B,3,H,W) in tensor
    if isinstance(image, np.ndarray):
        if image.ndim !=3 or image.shape[2] != 3:
            raise RuntimeError(f'invalid image shape {image.shape}')
        else:
            return image * std_np.reshape((1,1,3)) + mean_np.reshape((1,1,3))
    elif torch.is_tensor(image):
        if image.ndimension() !=4 or image.shape[1] != 3:
            raise RuntimeError(f'invalid image shape {image.shape}')
        else:
            return image * std.view(1,3,1,1).to(image.device) + mean.view(1,3,1,1).to(image.device)
    else:
        raise RuntimeError(f"unrecognizable image type {type(image)}")

# -----------------------------------------------------------------------------

class Trainer(BaseTrainer):
    def __init__(self, args, device):
        super().__init__(args)
        self.args = args
        self.device = device
        self.vis_view_ids = None

    def register_mesh_sampler(self):
        mesh_sampler_fname = join(self.directory_output, 'mesh_sampler.npz')
        if os.path.exists(mesh_sampler_fname):
            self.mesh_sampler = MeshSampler()
            self.mesh_sampler.load(mesh_sampler_fname)
        else:
            template_fname = self.args.template_fname
            if not os.path.exists(template_fname):
                raise RuntimeError('Template not found %s' % template_fname)
            template_mesh = Mesh(filename=template_fname)
            template_mesh.v[:] *= 1000
            mesh_dimension_list = self.args.number_sample_points
            self.mesh_sampler = MeshSampler(template_mesh, mesh_dimension_list, keep_boundary_adjacent=True)
            self.mesh_sampler.save(mesh_sampler_fname)

    def register_model(self):
        import models.model_aligner.prototypes.model_global_stage as models
        model = models.Model(args=self.args)
        model.initialize(init_method='normal')
        if self.args.pretrained_path:
            raise RuntimeError('Not yet tested if loading is good')
        model = model.to(self.device)
        self.model = torch.nn.DataParallel(model)

    def register_dataset(self, init_full_dataset=False, process_idx=0, world_size=1):
        ava_root = self.args.dataset_directory
        cam_angles = ast.literal_eval(self.args.camera_angles)
        subjects = ast.literal_eval(self.args.subjects) if self.args.subjects else list()
        frame_stride = self.args.frame_stride
        ava_height = self.args.height
        ava_width = self.args.width
        max_captures = self.args.max_captures

        nersemble_root = getattr(self.args, 'nersemble_dataset_directory', None)
        nersemble_subjects = ast.literal_eval(self.args.nersemble_subjects) if getattr(self.args, 'nersemble_subjects', None) else list()
        nersemble_frame_stride = getattr(self.args, 'nersemble_frame_stride', None)
        nersemble_height = getattr(self.args, 'nersemble_height', None)
        nersemble_width = getattr(self.args, 'nersemble_width', None)
        nersemble_max_captures = getattr(self.args, 'nersemble_max_captures', None)
        nersemble_camera_angles = ast.literal_eval(self.args.nersemble_camera_angles) if getattr(self.args, 'nersemble_camera_angles', None) else list()
        self.dataset_full = self.dataset_train = self.dataset_val = None
        self.dataloader_full = self.dataloader_train = self.dataloader_val = None

        if init_full_dataset:
            ava_dataset = AvaMultiCaptureDataset(
            root_path=ava_root, 
            camera_angles=cam_angles, 
            training=None,
            height=ava_height, width=ava_width,
            frame_stride=frame_stride,
            max_captures = max_captures,
            process_idx=process_idx, 
            world_size=world_size,
            deterministic_shuffle=False,
            require_verts=False,
            require_segmentation=False,
            only_subjects=subjects
            )
            datasets = [ava_dataset]
            if nersemble_root is not None:
                nersemble_dataset = NersembleMultiCaptureDataset(
                    root_path=nersemble_root,
                    camera_angles=nersemble_camera_angles,
                    training=None,
                    height=nersemble_height, width=nersemble_width,
                    frame_stride=nersemble_frame_stride,
                    max_captures = nersemble_max_captures,
                    process_idx=process_idx,
                    world_size=world_size,
                    deterministic_shuffle=False,
                    invalid_captures_path = 'assets/nersemble/invalid_captures.txt',
                    require_verts=False,
                    require_segmentation=False,
                    only_subjects=nersemble_subjects)
                datasets.append(nersemble_dataset)
            self.dataset_full = MultiDataset(datasets)
            self.dataloader_full = self.make_data_loader(self.dataset_full, cuda=True, shuffle=False, batch_size=1)
        else:
            self.dataset_train = AvaMultiCaptureDataset(
                root_path=ava_root, 
                camera_angles=cam_angles, 
                training=True,
                height=ava_height, width=ava_width,
                frame_stride=frame_stride,
                max_captures = max_captures,
                process_idx=process_idx,
                world_size=world_size,
                require_segmentation=False,
            )
            self.dataset_val = AvaMultiCaptureDataset(
                root_path=ava_root, 
                camera_angles=cam_angles, 
                training=False,
                height=ava_height, width=ava_width,
                frame_stride=frame_stride,
                deterministic_shuffle=True,
                max_captures = max_captures,
                process_idx=process_idx,
                world_size=world_size,
                require_segmentation=False,
            )

            self.dataloader_train = self.make_data_loader(self.dataset_train, cuda=True, shuffle=True, in_order=False, prefetch_factor=4)
            self.dataloader_val = self.make_data_loader(self.dataset_val, cuda=True, shuffle=False, num_workers=4)

    def register_losses(self):
        sample_mesh = self.mesh_sampler.get_mesh(-1)

        with open(self.args.vertex_group_path, 'r') as f:
            vertex_groups = json.load(f)
        for k, v in vertex_groups.items():
            vertex_groups[k] = np.array(v)
        n_points = sample_mesh.v.shape[0]
        
        # Point-to-point loss.
        vertex_group_weights = self.args.vertex_group_weights
        self._point_to_point_loss = point_to_point_loss.WeightedP2PLoss(
            num_points=n_points,
            vertex_groups=vertex_groups,
            group_weights=vertex_group_weights,
        )
        self._normal_loss = normal_loss.WeightedNormalLoss(
            num_points=n_points,
            faces=self.template_faces,
            vertex_groups=vertex_groups,
            group_weights=vertex_group_weights,
        )


        # self.points2surface_loss_function = PointToSurfaceLoss(gmo_sigma=self.args.gmo_sigma)
        # self.edge_loss_function = EdgeLoss( num_vertices=sample_mesh.v.shape[0], faces=sample_mesh.f, 
        #                                     vertex_masks=vertex_masks, mask_weights=self.args.edge_mask_weights,
        #                                     mesh_sampler=self.mesh_sampler)

    def feed_data(self, data):
        # suffix = '_augmented' if self.training else ''
        suffix = ''  # Disabled image augmentation
        if self.args.input_image_type == 'stereo_images':
            number_views = data['stereo_images'].shape[1]
            images = data[f'stereo_images{suffix}']
            camera_intrinsics = data[f'stereo_camera_intrinsics{suffix}']
            camera_extrinsics = data['stereo_camera_extrinsics']
            camera_distortions = data['stereo_camera_distortions']
        elif self.args.input_image_type == 'color_images':
            number_views = data['color_images'].shape[1]
            images = data[f'color_images{suffix}']
            camera_intrinsics = data[f'color_camera_intrinsics{suffix}']
            camera_extrinsics = data['color_camera_extrinsics']
            camera_distortions = data['color_camera_distortions']            
        else:
            raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

        if self.args.sample_views and self.model.training:
            views = get_subset_views(number_views, minimum_views=self.args.minimum_sample_views)
        else:
            views = np.arange(number_views)

        self.data = data
        self.inputs = {
            'images': Variable(images[:,views,...]).to(self.device),
            'camera_intrinsics': Variable(camera_intrinsics[:,views,...]).to(self.device),
            'camera_extrinsics': Variable(camera_extrinsics[:,views,...]).to(self.device),
            'camera_distortions': Variable(camera_distortions[:,views,...]).to(self.device)
        }

        self.target_vertices = data['v_registration'].to(self.device)

    def forward(self):
        random_grid = True if self.model.training else False
        self.global_points, _, self.global_grid, self.global_grid_origin, self.global_grid_scales = self.model(**self.inputs, random_grid=random_grid)

        # # visualize grid, prediction and gt

        # import einops
        # predicted_points = self.global_points
        # target_points = self.target_vertices
        # grid = self.global_grid
        # global_grid_preroi = torch.load('/tmp/global_grid.pt')

        # vis_pcs = dict(predicted_points=predicted_points[0].detach().cpu().numpy(),
        #                grid=einops.rearrange(grid[0].detach().cpu().numpy(), 'v c l1 l2 l3 -> (l1 l2 l3 v) c'),
        #                target_points = target_points[0].detach().cpu().numpy(),
        #                grid_preroi=einops.rearrange(global_grid_preroi[0].detach().cpu().numpy(), 'v c l1 l2 l3 -> (l1 l2 l3 v) c'),)
        # vis_3d_point_clouds(vis_pcs, 'demos/vis_pcs.html')
                       

        if self.global_step % self.args.print_frequency == 0:        
            print_memory(self.device, prefix='FW')

    def compute_losses(self):
        self.points_loss, self.normal_loss = 0.0, 0.0

        if self.args.weight_p2p > 0.0:
            self.points_loss = self._point_to_point_loss(self.global_points, self.target_vertices)
        if self.args.weight_normal > 0.0:
            self.normal_loss = self._normal_loss(self.global_points, self.target_vertices)
        # if self.args.weight_points2surface > 0.0:
        #     self.points2surface_loss = self.args.weight_points2surface*self._points2surface_loss()
        # if self.args.weight_edge_regularizer > 0.0:
        #     self.edge_regularizer_loss = self.args.weight_edge_regularizer*self._edge_regularizer()
        self.loss = self.args.weight_p2p*self.points_loss + self.args.weight_normal*self.normal_loss

    def backward(self):
        self.optimizer_model.zero_grad()
        self.loss.backward()
        if self.args.gradient_max_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.module.optimizable_parms(), max_norm=self.args.gradient_max_norm, norm_type=2)
        self.optimizer_model.step()
        self.scheduler_model.step()

    def run(self):
        batch_size = self.args.batch_size
        num_epoch = int(np.ceil(self.args.num_iterations / float(len(self.dataset_train)) * batch_size))
        start_epoch = int(self.global_step / float(len(self.dataset_train)) * batch_size)+1
        print("expect to run for %d epoches" % (num_epoch-start_epoch))

        for epoch_idx in range(start_epoch, num_epoch+1):
            np.random.seed() # reset seed
            print('************************')
            print('Epoch %d / %d' % (epoch_idx, num_epoch))
            print('************************')
            self.train_one_epoch()

    def save_uvmaps(self, outpath):
        n_samples = self.args.uv_n_samples
        for i, data in enumerate(tqdm.tqdm(self.dataloader_full, desc='Saving UVMaps', total=n_samples if n_samples != -1 else None)):
            if n_samples != -1 and i >= n_samples:
                break
            self.save_uvmap_batch(data, outpath)

    @torch.no_grad()
    def save_uvmap_batch(self, data, outdir, save_gt=False, save_verts=True):
        data = data_util.TempehBatch.from_match_batch(data_util.MatchBatch(data))
        ds_classes = [AvaSingleCaptureDataset if (not 'dataset_idx' in data) or (data['dataset_idx'][j] == 0) else NersembleSingleCaptureDataset for j in range(len(data['subject']))]
        out_subdirs = [Path(f'{outdir}/{"ava-256" if ds_classes[j]==AvaSingleCaptureDataset else "nersemble"}/{data["subject"][j]}/{data["sequence"][j]}/{data["frame"][j]}') for j in range(len(data['subject']))]
        out_paths_vert = list()
        out_paths_vis = list()
        out_paths_uv = list()
        out_paths_uv_gt = list()
        imgs = data['stereo_images']
        b, v = imgs.shape[:2]
        for ib in range(b):
          out_subdir = out_subdirs[ib]
          out_paths_vert.append(out_subdir/f"verts.npy")
          out_paths_vis.append(out_subdir/f"vis.jpg")
          out_paths_uv_ = list()
          out_paths_uv_gt_ = list()
          for iv in range(v):
            out_paths_uv_.append(out_subdir/f"uv_cam{int(data['camera'][ib][iv])}.png")
            out_paths_uv_gt_.append(out_subdir/f"uv_gt_cam{int(data['camera'][ib][iv])}.png")
          out_paths_uv.append(out_paths_uv_)
          if save_gt:
            out_paths_uv_gt.append(out_paths_uv_gt_)
            
        if all(p.exists() for p in out_paths_vert) and all([p.exists() for sublist in out_paths_uv for p in sublist]) and all([p.exists() for sublist in out_paths_uv_gt for p in sublist]):
          return


        self.set_eval()
        self.feed_data(data)
        self.forward()

        # save prediction: 
        pred_points = self.global_points
        data['v_pred'] = pred_points.cpu()
        uv_renders = uv_util.render_uvmaps_from_TEMPEH_sample(data, faces=self.template_faces, face_uv_coords=self.template_face_uv_coords,
                                                              out_height=self.args.uv_out_height, out_width=self.args.uv_out_width)
        if save_gt:
            uv_gt_renders = uv_util.render_uvmaps_from_TEMPEH_sample(data, faces=self.template_faces, face_uv_coords=self.template_face_uv_coords,
                                                              out_height=self.args.uv_out_height, out_width=self.args.uv_out_width, use_gtverts=True)
        
        # converting back from TEMPEH coordinate system to MATCH coordinate system
        pred_points = pred_points.detach().cpu().numpy()
        pred_points /= 1000.0
        B=len(pred_points)

        # converting to respective dataset coordinate system
        ds_scale_factor = np.stack([ds_cls.SCALE_FACTOR for ds_cls in ds_classes])
        ds_center = np.stack([ds_cls.SCENE_CENTER for ds_cls in ds_classes])
        ds_rotation = np.stack([ds_cls.SCENE_ROTATION for ds_cls in ds_classes])
        pred_points = data_utils.apply_inv_rotation_scale_center_to_points(pred_points, rotation=ds_rotation, scale=ds_scale_factor, center=ds_center)

        for ib in range(len(pred_points)):
            out_path = out_paths_vert[ib]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if save_verts:
                tmp_path = out_path.parent / out_path.name.replace(
                    ".npy", "_tmp.npy"
                )
                with file_helper.open_file(tmp_path, "wb") as f:
                    np.save(f, pred_points[ib],)
                file_helper.rename(tmp_path, out_path, overwrite=True)

            for iv in range(v):
                out_path = out_paths_uv[ib][iv]
                tmp_path = out_path.parent / out_path.name.replace(
                ".png", "_tmp.png"
                )
                Image.fromarray(np.round(uv_renders[ib, iv]*255).clip(0, 255).astype(np.uint8)).save(str(tmp_path))
                file_helper.rename(tmp_path, out_path, overwrite=True)

                if save_gt:
                    out_path = out_paths_uv_gt[ib][iv]
                    tmp_path = out_path.parent / out_path.name.replace(
                        ".png", "_tmp.png"
                    )
                    Image.fromarray(np.round(uv_gt_renders[ib, iv]*255).clip(0, 255).astype(np.uint8)).save(str(tmp_path))
                    file_helper.rename(tmp_path, out_path, overwrite=True)

          
            # save visualization
            nvis=5
            imgs_vis = (imgs[ib]*.5+.5)[:nvis]
            imgs_vis = torchvision.transforms.functional.resize(imgs_vis, uv_renders.shape[2:4])
            imgs_vis = einops.rearrange(imgs_vis, 'V C H W -> H (V W) C').cpu().numpy()
            uv_vis = einops.rearrange(uv_renders[ib, :nvis], 'V H W C -> H (V W) C')
            vis_img = np.concatenate([imgs_vis, uv_vis, (imgs_vis + uv_vis)*.5], axis=0)
            vis_img = Image.fromarray(np.round(vis_img*255).clip(0, 255).astype(np.uint8))
            h_vis = 512
            w_vis = int(vis_img.width / vis_img.height * h_vis)
            vis_img = vis_img.resize((w_vis, h_vis))
            vis_img.save(str(out_paths_vis[ib]))
            vis_img.close()

    def train_one_epoch(self):
        for data in self.dataloader_train:
            self.train_step(data)

    def train_step(self, data):
        self.set_train()
        data = data_util.TempehBatch.from_match_batch(data_util.MatchBatch(data))
        self.feed_data(data)
        self.forward()
        self.compute_losses()
        self.backward()
        lr = self.scheduler_model.get_last_lr()[0]

        if self.global_step % self.args.print_frequency == 0:        
            print('%s, step %d, total loss: %f, learning rate: %f' %(get_time_string(), self.global_step, to_numpy(self.loss), lr))
            self.tb_logger.add_scalar('Learning rate/train', lr, self.global_step)
            self.tb_logger.add_scalar('Total loss/train', to_numpy(self.loss), self.global_step)
            self.tb_logger.add_scalar('P2P loss/train', to_numpy(self.points_loss), self.global_step)
            self.tb_logger.add_scalar('Normal loss/train', to_numpy(self.normal_loss), self.global_step)
            # self.tb_logger.add_scalar('Edge regularizer loss/train', to_numpy(self.edge_regularizer_loss), self.global_step)     
            # self.tb_logger.add_scalar('Points2Surface distance/train', to_numpy(self._points2surface_loss(no_robustifier=True)), self.global_step)        
        if (self.global_step > 0) and (self.global_step % self.args.visualize_frequency == 0):
            self.visualize('train')
        if (self.global_step > 0) and (self.global_step % self.args.validate_frequency == 0):               
            self.validate()
        if (self.global_step > 0) and (self.global_step % self.args.save_frequency == 0):
            self.save_checkpoint()
        self.global_step += 1

    def validate(self):
        self.set_eval()
        with torch.no_grad():
            total_val_loss = []
            total_points_loss = []
            total_points2surface_loss = []
            total_edge_reg_loss = []
            total_points2surface_distance = []
            with torch.no_grad():
                for i, data in enumerate(self.dataloader_val):
                    data = data_util.TempehBatch.from_match_batch(data_util.MatchBatch(data))
                    self.feed_data(data)
                    self.forward()
                    self.compute_losses()
                    total_val_loss.append(to_numpy(self.loss))
                    total_points_loss.append(to_numpy(self.points_loss))
                    total_points2surface_loss.append(to_numpy(self.normal_loss))
                    # total_edge_reg_loss.append(to_numpy(self.edge_regularizer_loss))
                    # total_points2surface_distance.append(to_numpy(self._points2surface_loss(no_robustifier=True)))       
                    
                    if i>= self.args.validation_steps:
                        break     
            self.tb_logger.add_scalar('Total loss/validation', np.mean(np.array(total_val_loss)), self.global_step)
            self.tb_logger.add_scalar('P2P loss/validation', np.mean(np.array(total_points_loss)), self.global_step)
            self.tb_logger.add_scalar('Normal loss/validation', np.mean(np.array(total_points2surface_loss)), self.global_step)
            # self.tb_logger.add_scalar('Edge regularizer loss/validation', np.mean(np.array(total_edge_reg_loss)), self.global_step)                              
            # self.tb_logger.add_scalar('Points2Surface distance/validation', np.mean(np.array(total_points2surface_distance)), self.global_step)        
            self.visualize('val')

    def visualize(self, mode='train'):
        if self.vis_view_ids is None:
            if self.args.input_image_type == 'stereo_images':
                num_views = len(self.data['stereo_images'][0])     
            elif self.args.input_image_type == 'color_images':
                num_views = len(self.data['color_images'][0])     
            else:
                raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

            view_ids = np.arange(num_views)
            random.Random(7).shuffle(view_ids)
            self.vis_view_ids = view_ids[:6]
        
        with torch.no_grad():
            target_vertices = to_numpy(self.data['v_registration'][0])
            reconstructed_vertices = to_numpy(self.global_points[0])
            faces = to_numpy(self.data['f_registration'][0])

            vertex_distance = np.linalg.norm(target_vertices-reconstructed_vertices, axis=-1)
            vertex_colors = dist_to_rgb(vertex_distance, min_dist=0.0, max_dist=3.0)

            for view_id in self.vis_view_ids:
                if self.args.input_image_type == 'stereo_images':
                    if view_id >= self.data['stereo_images'][0].shape[0]:
                        continue
                    input_image = to_numpy(self.data['stereo_images'][0][view_id].permute(1,2,0))
                    camera_intrinsics = to_numpy(self.data['stereo_camera_intrinsics'][0][view_id])
                    camera_extrinsics = to_numpy(self.data['stereo_camera_extrinsics'][0][view_id])
                    radial_distortion = to_numpy(self.data['stereo_camera_distortions'][0][view_id])
                elif self.args.input_image_type == 'color_images':
                    if view_id >= self.data['color_images'][0].shape[0]:
                        continue
                    input_image = to_numpy(self.data['color_images'][0][view_id].permute(1,2,0))
                    camera_intrinsics = to_numpy(self.data['color_camera_intrinsics'][0][view_id])
                    camera_extrinsics = to_numpy(self.data['color_camera_extrinsics'][0][view_id])
                    radial_distortion = to_numpy(self.data['color_camera_distortions'][0][view_id]) 
                else:
                    raise RuntimeError( "Unrecognizable input_image_type option: %s" % ( self.args.input_image_type ) )

                if mode == 'train':
                    input_image = denormalize_image(input_image)
                else:
                    input_image = denormalize_image(input_image)
                input_image = (255*input_image).astype(np.uint8)

                camera_args = {
                    'camera_intrinsics': camera_intrinsics,
                    'camera_extrinsics': camera_extrinsics,
                    'radial_distortion': radial_distortion,
                    'frustum': {'near': 0.01, 'far': 3000.0},
                    'image_size': input_image.shape[:2]
                }

                target_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=None, **camera_args)
                reconstruction_rendering = render_mesh(vertices=reconstructed_vertices, faces=faces, vertex_colors=None, **camera_args)
                target_error_rendering = render_mesh(vertices=target_vertices, faces=faces, vertex_colors=vertex_colors, **camera_args)
                visualization = np.hstack((input_image, target_rendering, reconstruction_rendering, target_error_rendering)).transpose(2,0,1)
                self.tb_logger.add_image('%s/view_id_%02d' % (mode.capitalize(), view_id), visualization, self.global_step)

    def _points_reconstruction_loss(self):
        '''
        Vertex-to-vertex difference between the target meshes and the reconstructed points. 
        '''

        return self.points_loss_function(self.global_points, self.target_vertices)

    def _points2surface_loss(self, no_robustifier=False):
        '''
        Point-to-surface difference between the target scans and the surface spanned by the reconstructed points. 
        The point-to-surface loss computes for every vertex in the target scan the distance to the closest point in the 
        surface of the mesh defined by the reconstructed points. 
        '''

        # if 'v_scan' not in self.data:
        #     print("No scan vertices available")
        #     return 0.0
        target_vertices = self.target_vertices
        predicted_vertices = self.global_points
        predicted_faces = self.data['f_reg_global'][0].to(self.device)
        target_points = kaolin.ops.mesh.sample_points(target_vertices, predicted_faces, self.args.scan_vertex_count)[0]


        if not no_robustifier:
            return self.points2surface_loss_function(target_points, predicted_vertices, predicted_faces)
        else:
            with torch.no_grad():
                distances = compute_s2m_distance(target_points, predicted_vertices, predicted_faces)
                return distances.mean(-1).mean()

    def _edge_regularizer(self):
        return self.edge_loss_function(self.global_points, self.target_vertices)

# -----------------------------------------------------------------------------

def run(config_fname=''):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    parser.print_options()

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set trainer
    trainer = Trainer(args, device)
    trainer.initialize()
    trainer.run()

def vis_3d_point_clouds(named_point_clouds: dict, outpath: str):
    """
    Visualizes multiple 3D point clouds with different colors in a single 3D plot, and saves as HTML.

    Args:
        named_point_clouds: Dictionary mapping names to point clouds as np.ndarray of shape (N,3)
        outpath: Path to save the HTML visualization
    """
    fig = go.Figure()

    # Get a color palette large enough for all point clouds
    colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    
    for idx, (name, points) in enumerate(named_point_clouds.items()):
        color = colors[idx % num_colors]
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=name
        )
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Point Clouds',
        legend=dict(itemsizing='constant')
    )
    
    fig.write_html(outpath)
    print(f"Visualization saved to {outpath}")

def save_uvmaps(config_fname, outpath, process_idx=0, world_size=1):
    parser = TrainOptions()
    args = parser.parse(config_filename=config_fname)
    parser.print_options()

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % args.gpu)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # set trainer
    trainer = Trainer(args, device)
    trainer.initialize(init_full_dataset=True, process_idx=process_idx, world_size=world_size)
    trainer.save_uvmaps(outpath=outpath)

if __name__ == '__main__':
    run()
    print('Done')

