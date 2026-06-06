# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: wojciech.zielonka@tuebingen.mpg.de, wojciech.zielonka@tu-darmstadt.de

from collections import defaultdict
import pudb
import os
import torchvision as tv
from pathlib import Path
import cv2
import numpy as np
from pytorch3d.ops import knn_points
import torch as th
from tqdm import tqdm
from encoder.encoder import ResnetEncoder
from gaussians.losses import VGGPerceptualLoss, l1_loss, l2_loss, ssim
from lib.apperance.model import ApperanceModel
from itertools import combinations
from loguru import logger
import torch.nn.functional as F
from  torch.optim.lr_scheduler import MultiStepLR
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.model import RegressorModel, RegressorMode
from utils.face_detector import FaceDetector
from utils.general import build_loader, get_single, instantiate, to_device, to_tensor
from utils.geometry import AttrDict
from utils.renderer import Renderer
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms.functional as Ftv
from torchvision.utils import save_image, make_grid
from lib.F3DMM.masks.masking import Masking
from pytorch3d.transforms import matrix_to_quaternion, axis_angle_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
from torchvision.utils import save_image
import dqtorch
import torch
import copy 
import ffmpeg
import shutil
import einops
from fused_ssim import fused_ssim
from models.vae import kl_loss_stable
from lib.common import convert_flame_fits_to_dataset_flame_params



L1_criterion = th.nn.L1Loss()

def axis_angle_to_quaternion(axis_angles):
    rot_mats = axis_angle_to_matrix(axis_angles)
    quaternions = matrix_to_quaternion(rot_mats)
    return quaternions

def quaternion_to_axis_angle(quaternions):
    rot_mats = quaternion_to_matrix(quaternions)
    axis_angles = matrix_to_axis_angle(rot_mats)
    return axis_angles


def RT_to_dualquat(RT):
    '''
    
    Args:
        RT: (..., 4, 4)
    Returns:
        dual quat (..., 8)
    '''
    rotquat = dqtorch.matrix_to_quaternion(RT[..., :3, :3])
    dualquat = dqtorch.quaternion_translation_to_dual_quaternion(rotquat, RT[..., :3, -1])
    return torch.cat(dualquat, dim=-1)


def dualquat_to_RT(dquat):
    '''
    
    Args:
        dquat (..., 8)
    
    Returns: 
        rotation matrix (..., 4, 4)
    '''

    rotquat = dquat[..., :4]
    transquat = dquat[..., 4:]

    rotquat = rotquat / rotquat.norm(dim=-1, keepdim=True)
    rotquat, transquat = dqtorch.dual_quaternion_rectify((rotquat, transquat))
    rotquat, T = dqtorch.dual_quaternion_to_quaternion_translation((rotquat, transquat))
    R = dqtorch.quaternion_to_matrix(rotquat)
    RT = torch.cat((R, T[..., None]), dim=-1)
    RT = torch.cat((RT, torch.zeros_like(RT[..., :1, :])), dim=-2)
    RT[..., -1, -1] = 1

    return RT

class RegressorTrainer(ApperanceTrainer):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)

    def initialize(self):
        self.masking = Masking()
        self.model = RegressorModel(self.config, self.dataset)
        self.use_gtempeh_predictions = self.config.train.get('use_gtempeh_predictions', False)


        # Disable certain regions of GEM
        # See gem/masks/flame for available masks
        test_disable_regions = self.config.train.get("test_disable_regions", ["hair"])
        logger.info(f"Regions disabled for testing: {test_disable_regions}")
        masks = []
        for region in test_disable_regions:
            masks.append(self.load_mask(region)[0])

        # Apply the same mask which removes the neck gaussians
        neck_mask, _ = self.load_mask("neck", invert=True)
        if self.use_gtempeh_predictions:
            neck_mask = th.ones_like(neck_mask)
        masked_gaussians = neck_mask[self.get_tex_to_mesh()].bool()[:, 0]

        self.k_nearest = self.get_k_nearest(4, self.model.apperance.get_mean("geometry").detach())

        # Accumulate the mask
        mask = th.sum(th.stack(masks, dim=0), dim=0) > 0

        # Set the mask for which only mean from GEM will be used
        self.model.inactive_gem_mask = mask[self.get_tex_to_mesh()].bool()[:, 0][masked_gaussians]

        H, W = self.config.height, self.config.width
        self.bg = th.zeros([3, H, W]).cuda() if self.bg_color == "black" else th.ones([3, H, W]).cuda()
        self.vgg_loss = VGGPerceptualLoss().cuda()
        self.tb_writer = SummaryWriter(log_dir=self.config.train.tb_dir)
        self.renderer = Renderer(white_background=self.bg_color == "white").cuda()
        self.use_data_augmentation = self.config.train.get("use_data_augmentation", False)
        self.use_parts = self.config.train.get("use_parts", False)
        self.is_eval = False
        self.dataset.include_lbs = True
        self.model.refine_basis = True
        self.basis_init_iteration = self.config.train.get("basis_init_iteration", 40_000)
        self.current_sentence = ""
        self.neutral_frame_mean_init_iteration = self.config.train.get('neutral_frame_mean_init_iteration', 0)
        self.mean_init_iteration = self.config.train.get('mean_init_iteration', 0)
        self.mode = RegressorMode.TRAIN

        self.build_optimizable_orientantion()

        params_basis = [
            {"params": self.model.apperance.basis_parameters(), "lr": self.config.train.learning_rates.pca_basis, "name": "basis"},
            {"params": self.opt_orientaiton.parameters(), "lr": self.config.train.learning_rates.RT, "name": "RT"},
        ]
        if self.config.train.get('optimizable_means', False):
            params_basis.append({"params": self.model.apperance.means_parameters(), "lr": self.config.train.learning_rates.pca_mean, "name": "means"})

        self.coeff_residuals = None
        if self.config.train.get('optimizable_codes', False):
            self.coeff_residuals = th.nn.ParameterDict([(k, th.zeros(self.model.apperance.n)) for k in self.opt_orientaiton]).cuda()            
            params_basis.append({"params": self.coeff_residuals.parameters(), "lr": self.config.train.learning_rates.coeff_residuals, "name": "coeff_residuals"})

        params_regressor = [{"params": self.model.resnet.regressor.parameters(), "lr": self.config.train.learning_rates.regressor, "name": "regressor"}]

        self.basis_optimizer = th.optim.Adam(params=params_basis)
        self.mean_optimizer = th.optim.Adam(params=self.model.apperance.means_parameters(), lr=self.config.train.learning_rates.mean_init) if self.config.train.optimizable_means else None
        self.regressor_optimizer = instantiate(self.config.train.optimizer, params=params_regressor)
        self.regressor_scheduler = instantiate(self.config.train.lr_scheduler, optimizer=self.regressor_optimizer)

        # Current optimizer
        self.optimizer = self.basis_optimizer
        self.scheduler = None
        self.pred_codes = None

        self.original_means = self.model.apperance.means_parameters_flat().detach()

    def get_canonical_features(self, frame):
        return self.model.resnet.get_canonical_features(frame)
    

    def neutral_frame_init_mean(self, batch):
        for i in tqdm(range(self.neutral_frame_mean_init_iteration), desc='Initializing mean'):
            self.step(batch)


    def get_mean_frame(self):
        '''
        Returns index (int) of frame with the smallest deviation from the PCA mean
        '''
        return (self.model.apperance.gt_coeffs['all_full']**2).sum(dim=-1).argmin().detach().cpu().item()
    
    def get_tex_to_mesh(self):
        return self.model.apperance.get_tex_map()
        # flat_uv, mesh_obj = self._get_flat_uv_and_mesh()
        # trimesh_mesh = to_trimesh(mesh_obj)
        # uv_np = flat_uv.cpu().numpy()
        # vertex_ids = trimesh_mesh.kdtree.query(uv_np)[1]
        # return th.from_numpy(vertex_ids).cuda()

    def register(self, loss, info, name):
        prefix = "REFINE_" if self.model.refine_basis else ""
        info[prefix + name] = loss.item()
        return loss

    def build_optimizable_orientantion(self):
        default_dst = f"experiments/GEM/orientations/ROOT_quat/{self.config.capture_id}.ptk"
        dst = self.config.train.get('orientations_path', default_dst)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(dst):
            logger.info(f"ROOT_quat loaded from {dst}")
            pt = th.load(dst, weights_only=False)
            self.opt_orientaiton = th.nn.ParameterDict()
            for key in pt.keys():
                self.opt_orientaiton[key] = th.zeros(8)
            self.opt_orientaiton.load_state_dict(pt)
            self.opt_orientaiton.cuda()
            return

        frames = th.nn.ParameterDict().cuda()
        loader = build_loader(self.dataset, batch_size=5, num_workers=8, shuffle=False)
        for batch in tqdm(loader, desc='Building optimizable orientantion'):
            for i in range(len(batch["cam_idx"])):
                single = get_single(to_device(batch), i)
                if single["cam_idx"] != self.config.data.test_camera:
                    continue
                frame_id = str(single["frame"].item())
                root_RT = single['root_RT']
                frames[frame_id] = th.nn.Parameter(RT_to_dualquat(root_RT))

        self.opt_orientaiton = frames
        th.save(frames.state_dict(), dst)

    def train(self, training=True):
        self.is_eval = not training
        self.model.train(training)
        self.mode = RegressorMode.TRAIN if training else RegressorMode.VAL
        # if not training: 
        #     self.model.resnet.to_jit()


    def eval(self):
        self.train(False)

    def test(self):
        self.eval()
        self.mode = RegressorMode.TEST
        self.model.test()
    
    def cross(self):
        self.eval()
        self.mode = RegressorMode.CROSS
        self.model.cross()

    def set_mode(self, mode:RegressorMode):
        if mode == RegressorMode.TRAIN:
            self.train()
        elif mode == RegressorMode.VAL:
            self.eval()
        elif mode == RegressorMode.TEST:
            self.test()
        elif mode == RegressorMode.CROSS:
            self.cross()
        else:
            raise ValueError(f'Unsupported Mode {mode}')
        

    def load_state_dict(self, state):
        opt_params, model_params = state
        opt_dict_basis, opt_dict_orientaiton, opt_dict_regressor, scheduler_dict = opt_params
        self.basis_optimizer.load_state_dict(opt_dict_basis)
        self.regressor_optimizer.load_state_dict(opt_dict_regressor)
        self.regressor_scheduler.load_state_dict(scheduler_dict)
        self.model.load_state_dict(model_params)
        self.opt_orientaiton.load_state_dict(opt_dict_orientaiton)

    def state_dict(self):
        opt_params = (
            self.basis_optimizer.state_dict(),
            self.opt_orientaiton.state_dict(),
            self.regressor_optimizer.state_dict(),
            self.regressor_scheduler.state_dict(),
        )
        return (opt_params, self.model.state_dict())

    def laplacian_loss(self, name, points, idx):
        if name.lower() != "rotation":
            loss = self.geometry_laplacian_loss(points, idx)
        else:
            loss = self.rotation_laplacian_loss(points, idx)
        return loss

    def geometry_laplacian_loss(self, points, idx):
        neighbor_points = points[idx]
        mean_neighbors = neighbor_points.mean(dim=1)
        loss = th.mean((points - mean_neighbors).pow(2))
        return loss

    def rotation_laplacian_loss(self, quaternions, idx):
        axis_angles = quaternion_to_axis_angle(quaternions)  # shape: [N, 3]
        neighbor_axis_angles = axis_angles[idx]
        mean_neighbors = neighbor_axis_angles.mean(dim=1)  # shape: [N, 3]
        diff = axis_angles - mean_neighbors  # shape: [N, 3]
        loss = th.mean(diff.pow(2))
        return loss

    @th.no_grad()
    def inference(self, batch, identity_features=None):
        # Each sequence/exp can jump between cameras takes
        B = len(batch['image'])
        assert B == 1, 'Cannot reset running box if batch size != 1'
        if self.current_sentence != batch["exp"][0]:
            self.model.reset_running_bbox()
            self.current_sentence = batch["exp"][0]

        deca_pkg, app_pkg = self.model.predict(batch, identity_features=identity_features, is_warmup=self.is_warmup)
        if app_pkg is None:
            return None, None
        mesh_rendering = self.render_mesh(batch, app_pkg.mesh, mask=self.neck_mask)

        gt_image = batch["image"]
        B, C, H, W = gt_image.shape
        if self.bg.shape != gt_image.shape:
            self.bg = th.zeros([B, 3, H, W]).cuda() if self.bg_color == "black" else th.ones([B, 3, H, W]).cuda()
        alpha = batch["alpha"]
        if alpha != None:
            gt_image = gt_image * alpha + self.bg * (1 - alpha)
        cam_id = batch["cam_idx"]
        pred_image = app_pkg.pred_image
        pred_alpha = app_pkg.pred_alpha

        return AttrDict(
            {
                "gt_image": gt_image,
                "pred_image": pred_image,
                "pred_alpha": pred_alpha,
                "cam_id": cam_id,
                "mesh_rendering": mesh_rendering,
            }
        ), None

    def switch_optimization_stage(self):
        if (self.is_mean_init or self.is_neutral_frame_mean_init) and self.optimizer != self.mean_optimizer:
            logger.info("Switching to Mean Init Optimization.")
            self.optimizer = self.mean_optimizer
            self.scheduler = None
            return
        elif self.is_basis_init and self.optimizer != self.basis_optimizer:
            logger.info("Switching to Basis Init Optimization. Mean Init done!")
            self.optimizer = self.basis_optimizer
            self.scheduler = None
            return
        elif self.is_regressor_training and self.optimizer != self.regressor_optimizer:
            logger.info("Switching to Regressor Optimization. Refining Basis is done!")
            self.optimizer = self.regressor_optimizer
            self.scheduler = self.regressor_scheduler
            self.model.refine_basis = False

    
    @property
    def is_neutral_frame_mean_init(self):
        return self.curr_iter <= self.neutral_frame_mean_init_iteration

    @property
    def is_mean_init(self):
        return (self.neutral_frame_mean_init_iteration < self.curr_iter) and (self.curr_iter <= self.mean_init_iteration) 

    @property
    def is_basis_init(self):
        return (self.mean_init_iteration < self.curr_iter) and (self.curr_iter <= self.basis_init_iteration)
    
    @property
    def is_regressor_training(self):
        return self.basis_init_iteration < self.curr_iter
    
    @property
    def is_warmup(self):
        return not self.is_regressor_training
    
    @torch.no_grad()
    def save_val_predictions(self, eval_loader, name:str, mode:RegressorMode, identity_features=None):
        eval_steps = self.config.train.get('eval_n_samples', 20)
        if eval_steps is None:
            eval_steps = len(eval_loader)
        old_smoothen_bbox = self.model.resnet.smoothen_bbox  # disabling smooth bbox because evaluating random samples
        self.model.resnet.smoothen_bbox = False

        out_dir = Path(self.run_dir, 'val_predictions', name, f'{self.curr_iter:06d}')
        out_dir.mkdir(exist_ok=True, parents=True)
        for i, batch in enumerate(tqdm(eval_loader, total=min(eval_steps, len(eval_loader)), desc=name)):
            if i == eval_steps: 
                break
            batch = to_device(batch)
            loss, payload, info = self.get_loss(batch, force_payload=True, identity_features=identity_features)
            if loss is None:
                continue
            
            gt_img, pred_img = payload[:2]
            mask, loss_mask = payload[-2:]
            vis_img = torch.cat([gt_img, pred_img], dim=-1)
            B = len(pred_img)
            assert B == 1
            out_pred_path = out_dir / f'{i:06d}_pred.jpg'
            out_gt_path = out_dir / f'{i:06d}_gt.jpg'
            out_vis_path = out_dir / f'{i:06d}_vis.jpg'
            out_mask_path = out_dir / f'{i:06d}_mask.png'
            out_loss_mask_path = out_dir / f'{i:06d}_lossmask.png'
            save_image(pred_img.squeeze(0), out_pred_path)
            save_image(gt_img.squeeze(0), out_gt_path)
            save_image(vis_img.squeeze(0), out_vis_path)
            save_image(mask.squeeze(0), out_mask_path)
            if mode != RegressorMode.CROSS:
                save_image(loss_mask.squeeze(0), out_loss_mask_path)
        src = str(out_dir / '*_vis.jpg')
        dst = str(out_dir.with_suffix('.mp4'))
        outputs = ffmpeg.input(src, pattern_type='glob', framerate=10)
        ffmpeg.filter(
            outputs,
            'drawtext',
            fontfile='Arial.ttf',
            text='%{frame_num}',
            start_number=0,
            x='(w-tw)/2',
            y='h-(2*lh)',
            fontcolor='black',
            fontsize=20,
            box=1,
            boxcolor='white',
            boxborderw=5,
        ).output(dst).overwrite_output().run()
        self.model.resnet.smoothen_bbox = old_smoothen_bbox


    @torch.no_grad()
    def run_evaluation(self, eval_loader, name:str, identity_features=None, save_images=False):
        eval_steps = self.config.train.get('eval_n_samples', 20)
        n_vis = self.config.train.get('eval_n_vis', 5)
        eval_infos = defaultdict(list)
        eval_vises = []
        sample_counter = 0
        old_smoothen_bbox = self.model.resnet.smoothen_bbox  # disabling smooth bbox because evaluating random samples
        self.model.resnet.smoothen_bbox = False

        for batch in tqdm(eval_loader, total=min(eval_steps, len(eval_loader)), desc=name):
            batch = to_device(batch)
            loss, payload, info = self.get_loss(batch, force_payload=len(eval_vises)<n_vis or save_images, identity_features=identity_features)
            if loss is None:
                continue
            for k, v in info.items():
                eval_infos[k].append(v)
            if payload is not None:
                eval_vises.append(self.make_progress_image(payload))
            
            sample_counter += 1
            if sample_counter >= eval_steps-1:
                break
        eval_infos = dict([(k, np.mean(np.array(v))) for k, v in eval_infos.items()])
        eval_vises = th.cat(eval_vises[:n_vis], dim=-2)
        for key in eval_infos.keys():
                self.tb_writer.add_scalar(f"{name}/{key}", info[key], self.curr_iter)

        path = Path(self.run_dir, name, f"{self.curr_iter:06d}.jpg")
        path.parent.mkdir(exist_ok=True, parents=True)
        tv.utils.save_image(eval_vises, path)

        self.tb_writer.add_image(f'{name}/vis', tv.transforms.functional.resize(eval_vises, 1024), self.curr_iter)
        self.model.resnet.smoothen_bbox = old_smoothen_bbox

    @torch.no_grad()
    def make_video(self, loader, name:str, identity_features=None, fps=24, framestride=1):
        self.model.reset_running_bbox()
        
        # creating frames
        frame_counter = 0
        out_dir = Path(self.run_dir, name, f"{self.curr_iter:06d}_frames")
        out_path = Path(self.run_dir, name, f"{self.curr_iter:06d}.mp4")
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        for i, batch in enumerate(tqdm(loader, total=len(loader), desc='video_' + name)):
            assert len(batch['image']) == 1
            if i % framestride != 0: 
                continue
            batch = to_device(batch)
            vis, _ = self.inference(batch, identity_features=identity_features)
            if vis is None:
                continue
            payload = (vis.gt_image, vis.pred_image, vis.cam_id, vis.mesh_rendering, None, vis)
            img = self.make_progress_image(payload)
            frame_out_path = out_dir / f'{frame_counter:06d}.jpg'
            save_image(img, frame_out_path)
            frame_counter += 1            

        # writing video
        src = str(out_dir/'*.jpg')
        dst = str(out_path)
        outputs = ffmpeg.input(src, pattern_type="glob", r=fps)
        ffmpeg.filter(outputs, "pad", width="ceil(iw/2)*2", height="ceil(ih/2)*2").output(
            dst,
            pix_fmt="yuv420p",
            crf=25,
        ).overwrite_output().run()

        # delete frames
        shutil.rmtree(out_dir)


        
    def get_loss(self, batch, force_payload=False, identity_features=None):
        batch = copy.deepcopy(batch)
        B = batch["image"].shape[0]

        self.switch_optimization_stage()

        coeff_residuals = None
        if self.mode in [RegressorMode.TRAIN, RegressorMode.VAL]:
            root_dquat = torch.stack([self.opt_orientaiton[str(f.item())] for f in batch['frame']], dim=0)
            root_RT = dualquat_to_RT(root_dquat)
            batch['root_RT'] = root_RT
            coeff_residuals = torch.stack([self.coeff_residuals[str(f.item())] for f in batch['frame']], dim=0) if self.coeff_residuals is not None else None

        deca_pkg, app_pkg = self.model.predict(batch, self.is_warmup, code_residuals=coeff_residuals, identity_features=identity_features)

        if deca_pkg is None:
            return  None, None, None

        info = {}
        loss = 0.0
        loss_weights = dict(
            w_l1=0.3,
            w_ssim=1.,
            w_vgg=0.03,
            w_reg = 1e-3,
            w_codes = 0.005,
            w_jaw = 0.005,
            w_static_mouth_color = 0.,
            w_reg_mean_init = 0., 
            w_reg_mean_init_neutral_frame = 0.,
            w_lapl_geometry = 1000., # only active if use_parts
            w_basis_reg = 0.,
            w_kl = 0.,
        )
        config_loss_weights = self.config.train.get('loss_weights', {})
        for k in config_loss_weights:  # avoiding typos in config
            assert k in loss_weights  
        loss_weights.update(config_loss_weights)

        #### LOSSES ####

        # Photomertric loss

        #### GROUND TRUTH ####

        gt_image = batch["image"]
        alpha = batch["alpha"]

        #### PREDICTION ####

        pred_image = app_pkg.pred_image
        bg = einops.rearrange(app_pkg.bg_color, 'b c -> b c 1 1')
        gt_image = gt_image * alpha + bg * (1 - alpha)


        ### Masking 
        pred_image_masked = pred_image
        gt_image_masked = gt_image
        img_loss_mask = batch['img_loss_mask']

        if self.mode != RegressorMode.CROSS:
            pred_image_masked = pred_image_masked * img_loss_mask
            gt_image_masked = gt_image_masked * img_loss_mask

        #### LOSSES ####

        rgb_loss = l1_loss(pred_image_masked, gt_image_masked) 
        loss += self.register(rgb_loss, info, "L1") * loss_weights['w_l1']

        dssim = (1.0 - fused_ssim(pred_image_masked, gt_image_masked))
        loss += self.register(dssim, info, "D-SSIM") * loss_weights['w_ssim']

        if loss_weights['w_vgg']>0 or self.is_eval:
            vgg = self.vgg_loss(pred_image_masked, gt_image_masked) 
            loss += self.register(vgg, info, "VGG") * loss_weights['w_vgg']

        if self.config.train.get('orthogonalize_basis', True) and self.model.refine_basis and self.curr_iter % 1000 == 0:  
            self.model.apperance.make_orthagonal()

        ##### Regression loss #####
        if not self.model.refine_basis and self.mode in [RegressorMode.TRAIN, RegressorMode.VAL]:
            for k in deca_pkg.pred_codes.keys():
                reg = th.mean(deca_pkg.pred_codes[k] ** 2) 
                loss += self.register(reg, info, f"{k.upper()}_REG")* loss_weights['w_reg']
                l1_codes = L1_criterion(deca_pkg.pred_codes[k], deca_pkg.gt_codes[k]) 
                loss += self.register(l1_codes, info, f"{k.upper()}_LOSS")* loss_weights['w_codes']
            
            # jaw pose 
            l1_jaw = L1_criterion(deca_pkg.jaw_pose, batch['flame_params']['jaw_pose'])
            loss += self.register(l1_jaw, info, f"jaw_LOSS")* loss_weights['w_jaw']

        #### Static mouth color regularization loss #####
        if loss_weights['w_static_mouth_color'] > 0 and self.is_basis_init:
            apperance = self.model.apperance
            pca_color_components = apperance.unzip_all(dict(all=apperance.components['all_full'].reshape(apperance.n, -1, apperance.channels['all_full'])))['colors']  # (ncomponents, ngaussians, 3)
            pca_color_components_mouth = pca_color_components[:,self.masking.mouth()[apperance.tex_map_full.cpu()]]
            static_mouth_color_loss = torch.mean(pca_color_components_mouth**2)
            loss += self.register(static_mouth_color_loss, info, f"StaticMouthColor_REG") * loss_weights['w_static_mouth_color']

        # pca basis regularization
        basis_reg = torch.mean(self.model.apperance.basis_parameters_flat()**2)
        loss += self.register(basis_reg, info, f"Basis_REG") * loss_weights['w_basis_reg']

        #### Mean regularization loss
        if self.curr_iter <= self.mean_init_iteration:
            means = self.model.apperance.means_parameters_flat()
            loss_mean_reg = (((means - self.original_means)/self.original_means)**2).mean()
            if self.is_mean_init:
                w_loss_mean_reg = loss_weights['w_reg_mean_init']
            elif self.is_neutral_frame_mean_init:
                w_loss_mean_reg = loss_weights['w_reg_mean_init_neutral_frame']
            else:
                raise ValueError()
            loss += self.register(loss_mean_reg, info, f"OriginalMean_REG") * w_loss_mean_reg

        # KL Divergence
        if loss_weights['w_kl']>0:
            mus = deca_pkg.mus
            logstds = deca_pkg.logstds
            kl_loss = kl_loss_stable(mu=mus, logstd=logstds).mean()
            loss += self.register(kl_loss, info, f"KL") * loss_weights['w_kl']

        ##### Laplacian loss #####
        if self.use_parts:
            idxs = []
            k = 4
            for pkg in app_pkg:
                points_batch = pkg.gaussian["geometry"].unsqueeze(0)
                knn_out = knn_points(points_batch, points_batch, K=k)
                idxs.append(knn_out.idx[:, :, 1:][0])
            params = ["geometry"]
            lap_loss = sum(
                (sum(self.laplacian_loss(name, pkg.gaussian[name], idx) * loss_weights[f'w_lapl_{name}'] for name in params) / len(params))
                for idx, pkg in zip(idxs, app_pkg)
            ) / len(app_pkg)
            # Register laplacian loss
            loss += self.register(lap_loss, info, "LAP")

        ##### Final loss #####
        self.register(loss, info, "TOTAL")

        payload = None
        if force_payload or self.curr_iter % self.config.train.log_progress_n_steps == 0:
            mesh_rendering = self.render_mesh(batch, app_pkg.mesh)
            payload = (gt_image, pred_image.detach().clone(), batch['cam_idx'], mesh_rendering, alpha, img_loss_mask)

        return loss, payload, info
