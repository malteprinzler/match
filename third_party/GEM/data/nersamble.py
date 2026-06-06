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


import copy
import json
import os
from pathlib import Path
from PIL import Image
from loguru import logger
import cv2
import numpy as np
import torch as th
import trimesh
from data.base import BaseDataset, DatasetMode
from data.utils import load_obj, opengl_to_opencv, dilate
from lib.common import batchify_flame_params
from utils.geometry import AttrDict
from data.mesh import to_mesh, to_canonical
import torch
import glob 
import random
import re
import pudb

def get_orig_image_path(image_path):
    image_path = Path(image_path)
    frame_idx = int(image_path.name.split('_')[0])
    cam_idx = int(image_path.name.split('_')[1].split('.')[0])
    subject = image_path.parents[2].name
    sequence = image_path.parents[1].name
    orig_dir = Path('/is/cluster/mprinzler/gtempeh/experiments/gtempeh/distillation_datasets/finalava_uvres512_sapiens1b_mixed_780k')
    orig_seq_dir = orig_dir / subject / sequence
    orig_frame_dirs = [p for p in sorted(orig_seq_dir.iterdir()) if p.name.isnumeric()]
    orig_frame_dir = orig_frame_dirs[frame_idx]
    orig_image_path = orig_frame_dir/f'image_{cam_idx:02d}.jpg'
    return str(orig_image_path)


def to_dict(npz):
    params = {}
    for key in ["neck_pose", "jaw_pose", "eyes_pose", "shape", "expr", "static_offset", "rotation", "translation"]:
        val = np.array(npz[key]).astype(np.float32)
        if key != "shape":
            val = val[0:1]
        else:
            val = val[None]
        params[key] = val
    return params


def get_transforms(path):
    if not Path(path).exists():
        raise ValueError(f"Path {path} not found!")
    f = open(path)
    data = json.load(f)
    f.close()
    return data


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], list) and isinstance(value, list):
                dict1[key].extend(value)  # Concatenate lists
            elif isinstance(dict1[key], dict) and isinstance(value, dict):
                deep_merge(dict1[key], value)  # Recursive merge for dicts
            else:
                dict1[key] = value  # Override with new value
        else:
            dict1[key] = value  # Add new key

def get_face_feature_path(img_path:str, mode:DatasetMode)->str:
    suffix = '_train' if mode == DatasetMode.train else '_val'
    return img_path.replace('/images/', '/face_features/').replace('.jpg', f'{suffix}.npz').replace('.png', f'{suffix}.npz')

def get_backup_face_feature_paths(path:str):
        path = Path(path)  # pyright: ignore[reportAssignmentType]
        name = path.name
        dir = path.parent
        name_parse_re = r"^(\d+)_(\d{2})_(\w+)\.npz$"
        match = re.match(name_parse_re, name)
        frame_id, view_id, split = match.groups()
        candidate_pattern = str(dir/f'{frame_id}_*_{split}.npz')
        candidates = sorted(glob.glob(candidate_pattern))

        if len(candidates) == 0:  # ignore split
            candidate_pattern = str(dir/f'{frame_id}_*.npz')
            candidates = sorted(glob.glob(candidate_pattern))

        return candidates       


class NersembleDataset(BaseDataset):
    def __init__(self, config, custom_camera_list=None, mode=DatasetMode.train):
        super().__init__(config, custom_camera_list, mode)
        data = config.data
        self.imagepath = data.image
        self.maskpath = data.mask
        self.meshpath = data.mesh
        self.ds_rate = config.ds_rate
        self.config = config
        self.mode = mode
        self.mesh_topology = None
        self.custom_camera_list = custom_camera_list
        self.capture_id = config.capture_id
        self.root = str(Path(self.config.data_dir).parent)
        self.test_camera = config.data.test_camera
        self.load_face_features = config.data.get('load_pretrained_face_features', False)
        self.flame_fit_path = config.data.get('flame_fit_path', None)

        # texmean = np.asarray(Image.open(f"{str(Path(__file__).parent.parent)}/assets/textures/nersemble/flame_texture.png"), dtype=np.float32)
        # texmean = np.copy(np.flip(texmean, 0))
        # self.texmean = th.from_numpy(texmean).cuda().float().permute(2, 0, 1) / 255

        if mode == DatasetMode.train:
            self.transforms = get_transforms(config.data_dir + "/transforms_train.json")

        if mode == DatasetMode.validation:
            self.transforms = get_transforms(config.data_dir + "/transforms_val.json")

        if mode == DatasetMode.test:
            self.transforms = get_transforms(config.data_dir + "/transforms_test.json")

        join_configs = config.data.get("join_configs", False)
        if join_configs:
            self.transforms = self.join_configs(["transforms_train", "transforms_val"])

        self.create_cameras()
        self.create_topology()
        self.create_frames()
        self.set_identity()
        self.filter_frames()

    @staticmethod
    def gaussians_path_from_image_path(image_path:Path):
        gaussians_path = image_path.parents[1] / 'gaussians'/f'{image_path.name.split("_")[0]}.pt'
        return gaussians_path

    @staticmethod
    def cameras_path_from_image_path(image_path:Path):
        cameras_path = image_path.parents[1] / 'cameras'/f'{image_path.name.split("_")[0]}.json'
        return cameras_path

    @staticmethod
    def flame_path_from_image_path(image_path:Path)->Path:
        flame_path = image_path.parents[1] / 'flame_param' / f'{image_path.name.split("_")[0]}.npz'
        return flame_path

    def set_identity(self):
        self.identity_frame = None
        self.identity_img_path = None
        identity_frame = self.config.data.identity_frame
        for frame in self.join_configs(["transforms_train", "transforms_val", "transforms_test"])["frames"]:
            frame_id = str(frame["timestep_index"]).zfill(5)
            cam_id = str(frame["camera_index"]).zfill(2)
            if f"{frame_id}_{cam_id}" == identity_frame:
                self.identity_frame = frame
                self.identity_img_path = self.parse(frame["file_path"])
                break

        if self.identity_frame is None:
            logger.error(f"Idenity frame {identity_frame} was not found!")
            exit(-1)


    def join_configs(self, transforms):
        main = {}

        logger.info(f"Joining configs: {transforms}")

        for name in transforms:
            file_path = os.path.join(self.config.data_dir, f"{name}.json")
            transforms_data = load_json(file_path)
            
            if not main:
                main = copy.deepcopy(transforms_data)  # Initialize with first config
            else:
                deep_merge(main, transforms_data)  # Merge subsequent configs

        if "frames" in main and isinstance(main["frames"], list):
            main["frames"].sort(key=lambda x: x.get("file_path", ""))

        return main

    def filter_frames(self):
        if self.custom_camera_list is None:
            return
        filtered = []
        for frame in self.frame_list:
            cam_idx = str(frame["camera_index"]).zfill(2)
            if cam_idx in self.custom_camera_list:
                filtered.append(frame)

        self.frame_list = filtered
        self.frame_list.sort(key=lambda x: (x.get("timestep_index", float("inf")), x.get("camera_index", float("inf"))))

    def create_cameras(self):
        self.allcameras = list(map(lambda e: str(e).zfill(2), range(16)))

    def create_frames(self):
        frames = []
        # for frame in sorted(self.transforms["frames"], key=lambda x: x["file_path"]):
        for frame in self.transforms["frames"]:
            frames.append(frame)
        self.frame_list = frames

    def create_topology(self):
        obj = load_obj(f"assets/FLAME/head_template_mesh.obj")
        self.mesh_topology = obj

    def get_canonical_mesh(self):
        frame = self.frame_list[0]
        image_path = Path(self.parse(frame["file_path"]))
        flame_path = Path(self.flame_fit_path) / self.flame_path_from_image_path(image_path).relative_to(Path(image_path).parents[2])
        flame_params = np.load(flame_path)
        flame_params = dict([(k, np.array(v).astype(np.float32)[None]) for k, v in flame_params.items()])  # batchify
        mesh = to_canonical(flame_params)

        if self.config.train.get("canonical_mesh", False) and os.path.exists(self.config.train.canonical_mesh):
            mesh = trimesh.load(self.config.train.canonical_mesh, process=False)

        verts = mesh.vertices
        faces = mesh.faces

        logger.info(f"Loaded canonical mesh with {len(verts)} vertices and {len(faces)} faces")

        return th.from_numpy(verts).float().cuda(), th.from_numpy(faces).long().cuda()

    def parse(self, path):
        if Path(path).is_absolute():
            return path
        else:
            return self.root + "/" + path[2:]

    def get(self, idx):
        try:
            frame = self.frame_list[idx]
            return self.get_from_frame(frame)
        except Exception as e:
            logger.error(f"Error in get: {e}")
            return None  
    
    def get_from_frame(self, frame):
        cx = frame["cx"]
        cy = frame["cy"]
        fl_x = frame["fl_x"]
        fl_y = frame["fl_y"]
        h = frame["h"]
        w = frame["w"]

        sentnum = Path(frame["file_path"]).parent.parent.name
        timestep_index = frame["timestep_index"]
        cam = str(frame["camera_index"]).zfill(2)
        cam_id = frame["camera_index"]
        Rt = np.array(frame["transform_matrix"])

        Rt = opengl_to_opencv(Rt)

        K = np.eye(3)
        K[0, 2] = cx
        K[1, 2] = h-cy  # TODO FIX IN DATA SAVING!
        K[0, 0] = fl_x
        K[1, 1] = fl_y

        image_path = self.parse(frame["file_path"])
        alpha_path = image_path.replace("images", "fg_masks")
        seg_path = image_path.replace("images", "sapiens_seg").replace('.jpg', '.png')

        face_feature_path = None
        if self.load_face_features:
            face_feature_path = get_face_feature_path(image_path, self.mode)
            if not os.path.exists(face_feature_path):   # if face feature path doesnt exist, use other feature from same frame but different view
                candidates = get_backup_face_feature_paths(face_feature_path)
                if self.mode == DatasetMode.train:
                    random.shuffle(candidates)
                face_feature_path = candidates[0] if len(candidates)>0 else None

        image = self.read_image(image_path, cv2.INTER_CUBIC) / 255
        alpha = self.read_image(alpha_path, cv2.INTER_CUBIC)[0:1, :, :] / 255
        if os.path.exists(seg_path):
            seg = self.read_image(seg_path, cv2.INTER_NEAREST)
            torso_mask = (seg == 21).astype(np.float32)
            torso_mask = (dilate(torch.from_numpy(torso_mask[None, :1]), 2)>0).float().numpy()[0]
            img_loss_mask = 1-torso_mask
        else:
            img_loss_mask = np.ones_like(alpha)

        face_features = None
        if face_feature_path is not None:
            face_features = np.load(face_feature_path, allow_pickle=True)
            face_features = dict(face_features)
            face_features['smirk'] = face_features['smirk'].tolist()

        if self.flame_fit_path is not None:
            flame_path = Path(self.flame_fit_path) / self.flame_path_from_image_path(Path(image_path)).relative_to(Path(image_path).parents[2])
            flame_path = str(flame_path)

            flame_params = np.load(flame_path)
            flame_params = dict(flame_params)
            flame_params = batchify_flame_params(flame_params)

            R = th.from_numpy(flame_params["rotation"]).float()
            T = th.from_numpy(flame_params["translation"]).float()

            mesh, J, A, W = to_mesh(flame_params, R=R, T=T)

            R = A[0, 0, :3, :3]
            T = A[0, 0, :3, 3] + T

            root_Rt = np.eye(4)
            root_Rt[:3, :3] = R.numpy()
            root_Rt[:3, 3] = T.numpy()

            mesh, J, A, W = to_mesh(flame_params)
            verts = mesh.vertices

            flame_params = dict([(k, v[0]) for k, v in flame_params.items()])  # unbatchify flame params
            A = A.numpy()
            W = W.numpy()
        else:
            root_Rt = np.eye(4)
            verts = None
            A = None
            W = None
            flame_params = dict()
            flame_path = None

        pkg = {
            "K": K,
            "Rt": Rt,
            "cam_id": cam_id,
            "image": image,
            "alpha": alpha,
            "verts": verts,
            "root_Rt": root_Rt,
            "frame": timestep_index,
            "sentnum": sentnum,
            "cam": cam,
            "flame_path": flame_path,
            "image_path": image_path,
            "A": A,
            "W": W,
            'img_loss_mask': img_loss_mask,
            'flame_params': flame_params,
        }
        if face_features is not None:
            pkg['pretrained_face_features'] = face_features

        return AttrDict(pkg)

