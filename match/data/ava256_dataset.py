# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import io
import os
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
import einops
import numpy as np
import torch.utils.data
from PIL import Image
from plyfile import PlyData
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from zipfile import ZipFile, BadZipFile
from tqdm import tqdm
from zipp import Path as ZipPath
import cv2
import pathlib
from glob import glob
import io
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Union, TextIO, List
from zipfile import ZipFile
import pudb
import cv2
import copy
from collections import defaultdict
from match.utils import data_util, file_util, general_util, mesh_util
import collections
import re
import random
import hashlib

import einops
import numpy as np
import pandas as pd
from PIL import Image
from plyfile import PlyData
from zipp import Path as ZipPath
from match.data import base_dataset


T = TypeVar("T")

asset_dir = 'assets'

with open(f'{asset_dir}/ava256/validation_ids.json', 'r') as f:
    AVA_VAL_IDS = json.load(f)

class AvaCapture(base_dataset.MugsyCapture):
    """Unique identifier for a Mugsy capture"""

    def folder_name(self) -> str:
        return f"{self.mcd}--{self.mct}--{self.sid}"

AVA_NATIVE_HEIGHT = 4096
AVA_NATIVE_WIDTH = 2668
AVA_VERTEX_GROUP_PATH = f'{asset_dir}/ava256/vertex_groups.json'
with open(AVA_VERTEX_GROUP_PATH, 'r') as f:
    AVA_VERTEX_GROUPS = json.load(f)


class AvaSingleCaptureDataset(base_dataset.BaseSingleCaptureDataset):
    """
    Ava-256 dataset for a single capture

    Recommended size: height=786, width=512

    Args:
        capture: The unique identifier of this mugsy capture
    """

    SCALE_FACTOR = np.array(1./1000, dtype=np.float32)  # converting geometry from mm to m
    SCENE_CENTER = np.array([-0.0604,  0.0295,  0.9933], dtype=np.float32)


    def __init__(self, sapiens_segmentation_directory:str ='', **kwargs):
        super().__init__(**kwargs)        
        self.ava_original2cleaned_vert_mapping = np.load(f'{asset_dir}/ava256/original_2_cleaned_vertmapping.npy')
        self.sapiens_segmentation_directory = sapiens_segmentation_directory

    def get_neutral_seg_frame(self):
        neut_framelist = self.framelist.loc[self.framelist["seg_id"] == "EXP_neutral_peak"].values.tolist()
        if len(neut_framelist) == 0:
            logging.warning(f'Couldnt find neutral frame in sequence "EXP_neutral_peak" for subject {self.sid}, trying "EXP_eye_neutral" instead.')
            neut_framelist = self.framelist.loc[self.framelist["seg_id"] == "EXP_eye_neutral"].values.tolist()
        neut_framelist.sort()
        neut_seg, neut_frameid = neut_framelist[0]
        return neut_seg, neut_frameid

    def to_neutral_exp_dataset(self):
        neutral_self = copy.deepcopy(self)
        neut_seg, neut_frameid = neutral_self.get_neutral_seg_frame()
        neutral_self.framelist = neutral_self.framelist[(neutral_self.framelist['seg_id']==neut_seg) & (neutral_self.framelist['frame_id']==neut_frameid)]
        assert len(neutral_self) == 1
        return neutral_self
    
    def get_neutral_sample(self):
        neut_seg, neut_frameid = self.get_neutral_seg_frame()
        neutral_sample_idcs = np.where((self.framelist['seg_id']==neut_seg)&(self.framelist['frame_id'] == neut_frameid))[0].tolist()
        assert len(neutral_sample_idcs) == 1
        return self.__getitem__(neutral_sample_idcs[0])

    
    def get_frameid(self, idx:int)->str:
        return self.framelist.iloc[idx].frame_id

    def __len__(self):
        return len(self.framelist)

    @property
    def height_original(self)->int:
        return AVA_NATIVE_HEIGHT

    @property
    def width_original(self) ->int:
        return AVA_NATIVE_WIDTH

    @staticmethod
    def get_framelist(
        dataset_dir: Path, nframes: int = -1, frame_stride=1, skip_sequences=[]
    ) -> pd.DataFrame:
        """
        Load framelist
        """

        # Load frame list; ie, (segment, frame) pairs
        frame_list_path = f"{dataset_dir}/frame_list.csv"
        framelist = pd.read_csv(frame_list_path, dtype=str, sep=r",")
        framelist = framelist.iloc[::frame_stride]
        for s in skip_sequences:
            framelist = framelist[~framelist['seg_id'].str.startswith(s)]
        if nframes > 0:
            framelist = framelist.head(nframes)
        return framelist
    
    def load_metadata(self):
        self.framelist = self.get_framelist(dataset_dir = self.dir, nframes=self.nframes, frame_stride=self.frame_stride, skip_sequences=self.skip_sequences)
        self.full_framelist = self.get_framelist(dataset_dir = self.dir)
        self.krt_dict = self.load_camera_calibration()

        # check which cameras are available for which frames
        invalid_frame_cameras_path = self.dir / 'invalid_frame_cameras.json'
        if invalid_frame_cameras_path.exists():
            with open(str(invalid_frame_cameras_path), 'r') as f:
                self.invalid_frame_cameras.update(json.load(f))
        else:
            frames = list(self.full_framelist['frame_id'])
            for camera_id in self.krt_dict:
                camera_zip_path = self.camera_zip_path(camera_id=camera_id)
                with ZipFile(camera_zip_path) as f:
                    available_frames = [str(int(s.split('/')[1].strip('.avif'))) for s in f.namelist()]            
                for fid in frames:
                    if fid not in available_frames:
                        self.invalid_frame_cameras[fid].append(camera_id)
            invalid_frame_cameras_path.parent.mkdir(exist_ok=True, parents=True)
            with open(str(invalid_frame_cameras_path), 'w') as f:
                json.dump(dict(self.invalid_frame_cameras), f, indent='\t')

    def get_vertex_groups(self):
        return AVA_VERTEX_GROUPS

    def get_krt_dict(self):
        return self.krt_dict
    
    def load_camera_calibration(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load a KRT dictionary containing camera parameters
        Args:
            path: File path that contains the KRT information
        Returns:
            A dictionary with
                'intrin'
                'dist'
                'extrin'
        """

        with open(self.krt_file_path, "r") as f:
            camera_list = json.load(f)["KRT"]

        cameras = {}

        for item in camera_list:
            camera_name = item["cameraId"]
            file_path = self.dir / f'image/cam{camera_name}.zip'
            if not file_path.exists():
                continue

            RT = np.array(item["T"])
            RT = RT[:4, :3]
            RT = RT.T
            out = {
                "intrin": np.array(item["K"]).T,
                "extrin": RT,
                "dist": np.array([0, 0, 0, 0, 0.]), # images where already undistorted according to https://github.com/facebookresearch/ava-256/issues/11#issuecomment-2287556247
                "model": "radial-tangential",
            }

            cameras[camera_name] = out

        return cameras

    @property
    def krt_file_path(self):
        return self.dir / "camera_calibration.json"

    def read_image(self, path: ZipPath|Path, apply_gamma_correction=False) -> np.ndarray:
        """
        Returns: img as np.ndarray (H, W, C) 0...255
        """
        img_bytes = path.read_bytes()
        with io.BytesIO(img_bytes) as b:
            with Image.open(b) as img:
                img = np.asarray(img) 
        assert img.dtype == np.uint8


        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # Add channel dimension   

        if apply_gamma_correction:
            img = img.astype(np.float32) / 255.
            img = np.concatenate([ava256_linear2srgb(img[..., :3], dim=-1), img[..., 3:]], axis=-1)
            img = np.clip(np.round(img*255).astype(np.uint8), 0, 255)
        return img
    
    def load_headpose_and_verts(self, frame_id:str):
        '''
        
        Returns
            headpose (4,4)
            verts (N, 3)
        '''
        headpose = self.load_headpose(frame_id)
        verts = self.load_verts(frame_id)

        # verts are stored in head-centric coordinate system. Convert to world system
        verts = (headpose[:3, :3] @ verts.T).T + headpose[:3, -1][None]  

        return headpose, verts
    
    def load_headpose(self, frame_id:str) -> np.ndarray:
        '''
        Returns:
            headpose: (4,4)
        '''
        headpose = np.eye(4)
        with ZipFile(self.dir / "head_pose" / "head_pose.zip") as f:
            path = ZipPath(f, f"{int(frame_id):06d}.txt")
            headpose_bytes = path.read_bytes()
            with io.BytesIO(headpose_bytes) as b:
                headpose[:3] = np.loadtxt(b, dtype=np.float32)
        headpose = headpose.astype(np.float32)
        return headpose
    
    def get_sequenceid(self, frame_id:str)->str:
        return self.framelist.loc[self.framelist['frame_id'] == frame_id, 'seg_id'].iloc[0]

    def load_verts(self, frame_id: str) -> np.ndarray:
        zip_file_path = self.dir / "kinematic_tracking" / "registration_vertices.zip"
        if not zip_file_path.exists() and not self.require_verts:
            verts = np.zeros((7306, 3), dtype=np.float32)
            # logging.warning(f'Couldnt find kinematic tracking vertices under path {zip_file_path}. Returning zeroed vertices.')
        else:
            with ZipFile(self.dir / "kinematic_tracking" / "registration_vertices.zip") as f:
                path = ZipPath(f, f"{int(frame_id):06d}.ply")
                ply_bytes = path.read_bytes()
                with io.BytesIO(ply_bytes) as b:
                    # verts, _ = p3d.io.load_ply(io.BytesIO(ply_bytes))
                    plydata = PlyData.read(b)
            verts = plydata["vertex"].data
            verts = np.array([list(element) for element in verts])
        verts = verts[self.ava_original2cleaned_vert_mapping]
        return verts
    
    def camera_zip_path(self, camera_id: str):
        return self.dir / "image" / f"cam{camera_id}.zip"
    
    def get_image_path(self, frame_id: str, camera_id:str) -> ZipPath:
        return f"{str(self.camera_zip_path(camera_id=camera_id))}/cam{camera_id}/{int(frame_id):06d}.avif"
    

    def load_image(self, frame_id:str, camera_id: str)->np.ndarray:
        path = self.get_image_path(frame_id=frame_id, camera_id=camera_id)
        zippath = data_util.path_2_zippath(path)
        img = self.read_image(zippath, apply_gamma_correction=True)
        return img

    def check_image(self, frame_id:str, camera_id: str):
        try:
            path = self.get_image_path(frame_id=frame_id, camera_id=camera_id)
            zippath = data_util.path_2_zippath(path)
            if not zippath.exists():
                raise data_util.SkippableError(f'Image path for {self.dir}, frame: {frame_id}, camera: {camera_id} does not exist.')
        except IndentationError:
            pass


    def load_cleanplate(self, camera_id: str)->np.ndarray:
        with ZipFile(self.dir / "background_image/background_image.zip") as f:
            path = ZipPath(f,f"{camera_id}.avif")
            img = self.read_image(path, apply_gamma_correction=True)[..., :3]
        return img


    def load_sg(self, frame_id:str, camera_id: str)->np.ndarray|None:
        if self.sapiens_segmentation_directory:
            self.capture
            path = Path(os.path.join(self.sapiens_segmentation_directory, f'{self.capture.mcd}--{self.capture.mct}--{self.capture.sid}', f'cam{camera_id}', f'{int(frame_id):06d}.png'))
            img = self.read_image(path, apply_gamma_correction=False)
        else:
            img=None
        return img


    def load_fg_mask(self, frame_id:str, camera_id: str)->np.ndarray:
            with ZipFile(self.dir / "foreground_masks" / f"cam{camera_id}.zip") as f:
                path = ZipPath(f,f"{int(frame_id):06d}.avif",)
                img = self.read_image(path)[..., :1]
            return img

  
    @staticmethod
    def validate_camera(base_dir: str, camera_id: str):
        paths = [Path(base_dir) / "image" / f"cam{camera_id}.zip", 
                 Path(base_dir) / "segmentation_parts" / f"cam{camera_id}.zip",
                 Path(base_dir) / "foreground_masks" / f"cam{camera_id}.zip"]

                 
        try:
            for path in paths:
                with ZipFile(path, 'r') as zf:
                    if len(zf.namelist()) == 0:
                        logging.warning(f"Camera zip {path} is empty. Dropping camera {camera_id}")
                        return False

        except (FileNotFoundError, BadZipFile) as e:
            logging.warning(f"Couldn't load file {path}. Dropping camera {camera_id}: {e}")
            return False
        return True


class AvaMultiCaptureDataset(base_dataset.BaseMultiCaptureDataset):
    """
    Dataset with CA2 assets for multiple captures

    Args:
        captures: The unique identifiers of the mugsy captures in this dataset
    """
    SINGLE_CAPTURE_DATASET_CLS = AvaSingleCaptureDataset

    def to_neutral_exp_dataset(self):
        neutral_self = copy.deepcopy(self)

        for capture, single_capture_ds in neutral_self.single_capture_datasets.items():
            neutral_self.single_capture_datasets[capture] = single_capture_ds.to_neutral_exp_dataset()
        
        neutral_self.post_single_capture_dataset_loading_hook()
        return neutral_self
    
    def get_neutral_sample(self, subject_id:str):
        single_capture_ds = self.retrieve_single_capture_ds(subject=subject_id, sequence=None)
        return single_capture_ds.get_neutral_sample()


    def load_extra_camera_ids(self, path:str):

        extra_camera_ids = collections.defaultdict(list)

        with open(path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                pattern = re.compile(r"(?P<subject>[A-Z0-9]{6})/decoder/image/cam(?P<camera>\d+)\.zip/cam(?P=camera)/(?P<frame>\d+)\.avif")
                match = pattern.search(line)
                if match:
                    extra_camera_ids[f"{match.group('subject')}--{int(match.group('frame'))}"].append(match.group('camera'))
                else:
                    raise NotImplementedError()
        return extra_camera_ids
    
    def filter_extra_camera_ids_by_capture(self, extra_camera_ids, capture):
        if extra_camera_ids is None or len(extra_camera_ids.keys())==0:
            return None
        else:
            return dict([(k, v) for k, v in extra_camera_ids.items() if k.split('--')[0] == capture.sid])

    def retrieve_single_capture_ds(self, subject:str, sequence:str):
        capture = [c for c in self.captures if c.sid == subject][0]
        return self.single_capture_datasets[capture]

    @staticmethod
    def folder_parser(base_dir: Path, training:bool|None = None) -> Tuple[List[AvaCapture], List[Path]]:
        """
        Args:
            training: specifies split, if True: training split, if False: validation split, if None: all
        """
        captures = []
        dirs = []
        captures_paths = [p for p in sorted(glob(f"{base_dir}/*")) if file_util.is_directory(p)]

        for capture in tqdm(captures_paths):
            name = Path(capture).name
            mcd, mct, sid = name.split("--")
            if training is not None and training:
                if sid in AVA_VAL_IDS:
                    continue
            elif training is not None and not training:
                if sid not in AVA_VAL_IDS:
                    continue
            capture = AvaCapture(mcd=mcd, mct=mct, sid=sid)
            captures.append(capture)
            capture_dir = f"{base_dir}/{capture.folder_name()}/decoder"
            dirs.append(capture_dir)
        return captures, dirs

    def get_texture_norm_stats(self) -> Tuple[np.ndarray, float]:
        """
        Calculate the texture mean and variance across all subdatasets.
        Technically wrong since we just compute the mean of the means, but it's good enough
        """
        N = len(self.single_capture_datasets)

        # Mean
        texmean = None
        for capture, dataset in self.single_capture_datasets.items():
            if texmean is None:
                texmean = dataset.texmean.copy()
            else:
                texmean += dataset.texmean
        texmean /= N

        # Stdev
        if N == 1:
            # TODO(julieta) probably wrong?!
            texvar = np.mean((texmean - np.mean(texmean, axis=0, keepdims=True)) ** 2)
        else:
            texvar = 0.0
            for capture, dataset in self.single_capture_datasets.items():
                texvar += np.sum((dataset.texmean - texmean) ** 2)
            texvar /= texmean.size * N

        return texmean, math.sqrt(texvar)

    def get_mesh_vert_stats(self) -> Tuple[np.ndarray, float]:
        """
        Calculate the mesh mean and variance across all subdatasets
        """
        N = len(self.single_capture_datasets)

        # Mean
        vertmean = None
        for capture, dataset in self.single_capture_datasets.items():
            if vertmean is None:
                vertmean = dataset.vertmean.copy()
            else:
                vertmean += dataset.vertmean
        vertmean /= N

        # Stdev
        vertvar, vertvar_mean = 0.0, 0.0
        for capture, dataset in self.single_capture_datasets.items():
            vertvar += np.sum((dataset.vertmean - vertmean) ** 2)
            vertvar_mean += dataset.vertstd**2
        vertvar /= vertmean.size * N
        vertvar += vertvar_mean / N

        return vertmean, math.sqrt(vertvar)


def ava256_linear2srgb(img: Union[np.ndarray, torch.Tensor], dim: int = -1) -> Union[np.ndarray, torch.Tensor]:
    """
    Parameters
    ----------
        img: Image in linear sRGB space. Values should be in [0, 1]
        dim: Which dimension the color channel is

    Returns
    -------
        image in non-linear sRGB space with white balancing and gamma correction applied
    """

    if dim == -1:
        dim = len(img.shape) - 1
    assert img.shape[dim] == 3

    is_numpy = isinstance(img, np.ndarray)

    shape = [3 if i == dim else 1 for i in range(len(img.shape))]
    gamma = 1.5254
    black = [4.4 / 255, 3.1 / 255, 4.2 / 255]
    scale = 1.0 / 1.1059
    color_scale = [1.279545, 1.1059, 1.6]

    if is_numpy:
        color_scale = np.array(color_scale, dtype=np.float32).reshape(shape)
        black = np.array(black, dtype=np.float32).reshape(shape)
    else:
        color_scale = torch.tensor(color_scale, dtype=torch.float32, device=img.device).reshape(shape)
        black = torch.tensor(black, dtype=torch.float32, device=img.device).reshape(shape)

    img = (img * (color_scale * (scale / (1 - black))) - (black * (scale / (1 - black))))

    # img = img * color_scale
    # img = (scale / (1 - black)) * (img - black)

    if is_numpy:
        return np.clip(np.power(np.clip(img, a_min=1e-6, a_max=None), 1.0 / gamma), a_min=0.0, a_max=1.0)
    else:
        return torch.clamp(img.clamp(min=1e-6).pow(1.0 / gamma), min=0.0, max=1.0)



class AvaDataLoader(base_dataset.ParentDataLoader):
    DATASET_CLS = AvaMultiCaptureDataset
