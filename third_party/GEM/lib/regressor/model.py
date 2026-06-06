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


import torch as th
from collections import deque
from encoder.encoder import ResnetEncoder
from lib.common import convert_flame_fits_to_dataset_flame_params, pose_gaussians
from lib.F3DMM.FLAME2020.flame import FLAME
from lib.F3DMM.FLAME2023.flame import FLAME as FLAME23
from lib.apperance.model import ApperanceModel
from lib.apperance.pca_gaussian import PCApperance
from lib.base_model import BaseModel
from gaussians.renderer import splat
from lib.common import Mesh
from loguru import logger
from utils.geometry import (
    AttrDict,
)
import pudb
import copy
import enum

class RegressorMode(enum.Enum):
    TRAIN = enum.auto()  # Training
    VAL = enum.auto()    # Evaluation on seen expressions
    TEST = enum.auto()   # Evaluation on unseen expressions
    CROSS = enum.auto()  # Cross reenactment

class RegressorModel(BaseModel):
    def __init__(self, config, dataset) -> None:
        super().__init__(config, dataset)
        try:
            self.apperance = PCApperance(config).cuda()
        except Exception as e:
            logger.warning(f"Failed to create PCA appearance model, using None: {e}")
            self.apperance = None
        self.resnet = ResnetEncoder(self.apperance.n + 3 if self.apperance is not None else None, config, dataset).cuda()
        self.flame = FLAME().cuda().eval()  # for deca stuff 
        self.flame23 = FLAME23().cuda().eval()  # for posing/unposing
        self.is_eval = False
        self.refine_basis = False
        self.inactive_gem_mask = None
        self.mode = RegressorMode.TRAIN

    def train(self, training=True):
        self.is_eval = not training
        self.resnet.train(training)
        # self.resnet.to_jit()
        if training: 
            self.mode = RegressorMode.TRAIN
        else:
            self.mode = RegressorMode.VAL
        

    def eval(self):
        self.train(False)       

    def test(self):
        self.train(False)
        self.mode = RegressorMode.TEST

    def cross(self):
        self.train(False)
        self.mode = RegressorMode.CROSS 

    def reset_running_bbox(self):
        self.resnet.current_bbox = None
        self.resnet.running_window = deque(maxlen=self.resnet.windows_size)

    def parameters(self):
        return [self.resnet.parameters(), self.apperance.parameters()]

    def count_parameters(self):
        for model_name in ["apperance", "resnet"]:
            if hasattr(self, model_name):
                model = getattr(self, model_name)
                n = sum(p.numel() for p in model.parameters() if p.requires_grad)
                logger.info(f"{str(type(model).__name__).ljust(20, ' ')} parameters={n}")

    def create(self):
        pass

    def load_state_dict(self, state):
        resnet, apperance = state
        self.resnet.load_state_dict(resnet)
        self.apperance.load_state_dict(apperance)

    def state_dict(self):
        return (self.resnet.state_dict(), self.apperance.state_dict())

    def get_opt_params(self):
        return self.resnet.parameters()

    def step(self, curr_iter):
        pass

    def flame_mesh(self, codedict):
        pose = codedict["pose"]
        pose[:, :3] = 0
        verts, _, _ = self.flame(shape_params=codedict["shape"] * 0, expression_params=codedict["exp"], pose_params=codedict["pose"])
        return verts

    def splat(self, batch, results, to_canonical=False):
        return splat(batch=batch, results=results, bg_color='white' if self.is_eval else self.bg_color , to_canonical=to_canonical, twoDgs=self.config.train.get('twoDgs', False))

    def to_gaussian_maps(self, results):
        with th.no_grad():
            colors = results.apperance.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            means3D = results.geometry.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            opacity = results.opacity.reshape(1, self.uv_size, self.uv_size, 1).permute(0, 3, 1, 2)
            scales = results.scales.reshape(1, self.uv_size, self.uv_size, 3).permute(0, 3, 1, 2)
            rotation = results.rotation.reshape(1, self.uv_size, self.uv_size, 4).permute(0, 3, 1, 2)

            maps = {
                "scales": scales,
                "rotation": rotation,
                "position": means3D,
                "rgb": colors,
                "opacity": opacity,
            }

        return AttrDict(maps)

    def parse(self, pca):
        preds = {}
        n = 0
        t = self.apperance.n 
        for key in self.apperance.masks:
            preds[key] = pca[:, n : n + t]
            n += t

        return self.apperance.to_coeffs(preds, is_batch=True)

    def slice(self, pred_codes, b):
        preds = {}
        for k in pred_codes.keys():
            preds[k] = pred_codes[k][b]
        return preds

    def stack(self, list_dict):
        merged = {}
        for dict in list_dict:
            if not dict:
                continue
            for k in dict.keys():
                if k not in merged:
                    merged[k] = []
                merged[k].append(dict[k])

        for k in merged.keys():
            merged[k] = th.stack(merged[k])

        return merged

    def enable_region(self, coeffs, name):
        for k in coeffs.keys():
            if name in k:
                continue
            coeffs[k] *= 0.0
        return coeffs

    def predict(self, batch, is_warmup=False, code_residuals=None, identity_features=None):
        gt_codes = None

        # gt_R = matrix_to_axis_angle(single["root_RT"][:3, :3])
        # gt_T = single["root_RT"][:3, 3]

        N = self.pca_n_components

        #### PREDCITION ####
        preds = self.resnet(batch, identity_features=identity_features)
        if preds is None:
            return None, None

        raw_codes = preds.pca
        paresed_preds = self.parse(raw_codes)

        if code_residuals is not None:
            code_residuals = self.parse(code_residuals)

        deca = {}
        deca["pred_codes"] = paresed_preds
        deca["raw_codes"] = raw_codes
        deca['mus'] = preds.mus
        deca['logstds'] = preds.logstds
        deca['jaw_pose'] = preds.jaw_pose

        pred_codes = paresed_preds

        gt_codes = None
        if self.mode in [RegressorMode.TRAIN, RegressorMode.VAL]:
            gt_codes = self.apperance.get(batch["frame"])

        if code_residuals is not None and gt_codes is not None:
            for k in gt_codes:
                gt_codes[k] = gt_codes[k] + code_residuals[k]

        # if is_warmup and not self.is_eval:  # TODO
        if is_warmup and gt_codes is not None:
            pred_codes = gt_codes

        deca["gt_codes"] = gt_codes

        vertices = self.flame_mesh(preds.deca)
        # mesh = Mesh(single["geom_vertices"].float(), single["geom_faces"].long())
        mesh = Mesh(vertices, batch["geom_faces"][0].long())

        #### DECA DEBUG ####
        # frame_id = single["frame"].item()
        # camera_id = single["cam_idx"]
        # gt_image = single["image"]
        # alpha = single["alpha"]
        # gt_image = gt_image * alpha + (1 - alpha)
        # Path("debug").mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(f"debug/{frame_id}_{camera_id}_cropped.png", preds.deca_input[0][0].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 255)
        # cv2.imwrite(f"debug/{frame_id}_{camera_id}_input.png", gt_image.permute(1, 2, 0)[:, :, [2, 1, 0]].cpu().numpy() * 255)
        # trimesh.Trimesh(vertices.detach().cpu().numpy(), self.flame.faces_tensor.detach().cpu().numpy()).export(f"debug/{frame_id}_{camera_id}.ply")

        masking = None
        if self.is_eval:
            masking = {
                # For multipart regression
                "keys": ["scalp", "neck"],
                # For inactive regions
                "mask": self.inactive_gem_mask
            }

        # pred_codes = self.enable_region(pred_codes, "mouth")

        results = self.apperance.inverse_transform(pred_codes, masking=masking)
        flame_params = copy.copy(batch['flame_params'])
        if not is_warmup:
            flame_params['jaw_pose'] = deca['jaw_pose']
        flame_params = convert_flame_fits_to_dataset_flame_params(flame_params, inverse=True)
        results = AttrDict(pose_gaussians(results, flame_params, self.flame23, results.tex_map))

        pred_image, render_pkg, pred_alpha, bg_color = self.splat(batch, results, to_canonical=True)

        # For visualziation
        render_pkg["pred_image"] = pred_image
        render_pkg["pred_alpha"] = pred_alpha
        render_pkg["mesh"] = mesh
        render_pkg["gaussian"] = results
        render_pkg["n_gaussian"] = self.uv_size**2
        render_pkg["bg_color"] = bg_color

        

        return AttrDict(deca), render_pkg
