# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
import numpy as np
from PIL import Image
from torch import multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import cv2
import pathlib
from glob import glob
import io
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Union, TextIO, List
import pudb
import cv2
from match.utils import file_util, mesh_util, geo_util
from match.three_dmm.flame import FlameHead
from match.three_dmm.lbs import batch_rodrigues
import torch



import numpy as np
from PIL import Image
from match.data import base_dataset
from match.utils import file_util

class NersembleCapture(base_dataset.MugsyCapture):
    def __init__(
        self,
        path: str,  # Subject ID, three letters and three numbers, eg `avw368`
    ):
        sid = file_util.Path(path).parent.name
        mcd = file_util.Path(path).name
        super().__init__(mcd=mcd, mct='', sid=sid)
    
    def folder_name(self) -> str:
        return f"{self.sid}/{self.mcd}"
    



# mp.set_start_method("spawn", force=True)

T = TypeVar("T")

asset_dir = 'assets'

with open(f'{asset_dir}/nersemble/validation_ids.json', 'r') as f:
    NERSEMBLE_VAL_IDS = json.load(f)


NERSEMBLE_NATIVE_HEIGHT = 802
NERSEMBLE_NATIVE_WIDTH = 550
NERSEMBLE_VERTEX_GROUP_PATH = f'{asset_dir}/flame/vertex_groups.json'
with open(NERSEMBLE_VERTEX_GROUP_PATH, 'r') as f:
    NERSEMBLE_VERTEX_GROUPS = json.load(f)

FLAME = FlameHead(shape_params=300, expr_params=100).cpu().share_memory()

class NersembleSingleCaptureDataset(base_dataset.BaseSingleCaptureDataset):
    """
    Dataset with Mugsy assets for a single capture

    recommended width=550, height=802,


    Args:
        capture: The unique identifier of this mugsy capture
    """

    SCENE_ROTATION = np.array([[1., 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    SCENE_CENTER = np.array([0., 0., 0.], dtype=np.float32)
    SCALE_FACTOR = np.array(1., dtype=np.float32)


    def __init__(self, **kwargs):
        super().__init__(**kwargs)    
        self.sequence_id = self.dir.name

    def __len__(self):
        return len(self.frame_ids)

    @property
    def height_original(self):
        return NERSEMBLE_NATIVE_HEIGHT
    
    @property
    def width_original(self):
        return NERSEMBLE_NATIVE_WIDTH

    @property
    def invalid_cameras_file_path(self) -> file_util.Path:
        return self.dir / 'invalid_cameras.json'

    def get_vertex_groups(self):
        return NERSEMBLE_VERTEX_GROUPS

    @staticmethod
    def get_transforms(
        dataset_dir: Path,
    ) -> dict:
        """
        """

        # Load frame list; ie, (segment, frame) pairs
        frame_list_path = file_util.Path(f"{dataset_dir}/transforms.json")
        with file_util.open_file(frame_list_path, 'r') as f:
            transforms = json.load(f)
        return transforms
    
    def load_metadata(self):
        self.transforms = self.get_transforms(self.dir)



        def _get_camera_parameters(frame_info:dict):
            pose = np.array(frame_info["transform_matrix"])
            Rt = geo_util.invert_c2w(pose)
            Rt[[2, 1]] *= -1  # opengl to opencv

            K = np.eye(3, dtype=np.float32)
            K[0,0]=frame_info['fl_x']
            K[1,1]=frame_info['fl_y']
            K[0,2]=frame_info['cx']
            K[1,2]=frame_info['cy']

            dist = np.zeros(5, dtype=np.float32)
        
            camera_id = frame_info['camera_id']
            camera_idx = frame_info['camera_index']
            h = frame_info['h']
            w = frame_info['w']
            assert h == self.height_original
            assert w == self.width_original

            return {'camera_id': camera_id, 'intrin': K, 'extrin': Rt, 'dist': dist, 'model': 'radial-tangential', 'camera_idx': camera_idx}
        
        
        cameras = {}
        camera_id2idx = {}
        frame_ids = set()
        frame_infos = self.transforms['frames']
        for frame_info in frame_infos: 
            frame_ids.add(frame_info['timestep_id'])
            cam_info = _get_camera_parameters(frame_info)
            camera_id = cam_info.pop('camera_id')
            camera_idx = cam_info.pop('camera_idx')
            if camera_id in cameras:
                # check consistency of camera parameters across frames
                for k, v in cam_info.items():
                    v_comp = cameras[camera_id][k]
                    if isinstance(v, np.ndarray):
                        assert np.all(v == v_comp), f'Camera parameter {k} didnt match for camera {camera_id}:\n{v}, {v_comp}'
                    else:
                        assert v == v_comp, f'Camera parameter {k} didnt match for camera {camera_id}:\n{v}, {v_comp}'
            else:
                cameras[camera_id] = cam_info      

            if camera_id in camera_id2idx:
                assert camera_id2idx[camera_id] == camera_idx
            else:
                camera_id2idx[camera_id] = camera_idx

        
        self.krt_dict = cameras
        frame_ids = sorted(frame_ids)
        frame_ids = frame_ids[::self.frame_stride]
        if self.nframes > 0:
            frame_ids = frame_ids[:self.nframes]
        self.frame_ids = frame_ids
        self.camera_id2idx = camera_id2idx
    
    def get_frameid(self, idx:int)->str:
        return self.frame_ids[idx]

    def get_krt_dict(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Args:
            path: File path that contains the KRT information
        Returns:
            A dictionary with
                'intrin'
                'dist'
                'extrin'
        """

        return self.krt_dict
        

    def read_image(self, path: Path, apply_gamma_correction=False) -> np.ndarray:
        """
        Returns: img as np.ndarray (H, W, C) 0...255
        """
        img_bytes = path.read_bytes()
        with io.BytesIO(img_bytes) as b:
            with Image.open(b) as img:
                img = np.array(img)
        assert img.dtype == np.uint8


        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # Add channel dimension   

        if apply_gamma_correction:
            raise NotImplementedError('Gamma correction not implemented')
        return img
    
    def load_headpose_and_verts(self, frame_id:str) -> Tuple[np.ndarray, np.ndarray]:
        flame_param_path = self.dir / f"flame_param/{int(frame_id):05d}.npz"
        flame_params = np.load(flame_param_path)
        verts = self.forward_flame(flame_params)  # (N, 3)
        translation = flame_params['translation'][0]  # (3,)
        rotation = batch_rodrigues(torch.from_numpy(flame_params['rotation'])).numpy()[0]  # (3,3)
        headpose = np.eye(4, dtype=np.float32)
        headpose[:3,:3] = rotation
        headpose[:3, -1] = translation

        # ###
        # # visualization of head pose and verts:
        # ###
        # axis_length = 1.
        # output_file = 'demos/headpose_and_verts.html'

        # fig = go.Figure()

        # # Scatter plot of vertices
        # fig.add_trace(go.Scatter3d(
        #     x=verts[:, 0],
        #     y=verts[:, 1],
        #     z=verts[:, 2],
        #     mode='markers',
        #     marker=dict(size=2, color='black'),
        #     name='Vertices'
        # ))

        # # Define head pose axes
        # colors = ['red', 'green', 'blue']

        # for i in range(3):
        #     # Transform axis by rotation
        #     axis_dir = rotation[:, i] * axis_length

        #     # Add cone for axis
        #     fig.add_trace(go.Cone(
        #         x=[headpose[0,-1]],
        #         y=[headpose[1,-1]],
        #         z=[headpose[2,-1]],
        #         u=[headpose[0,i]],
        #         v=[headpose[1,i]],
        #         w=[headpose[2,i]],
        #         colorscale=[[0, colors[i]], [1, colors[i]]],
        #         showscale=False,
        #         sizemode='absolute',
        #         sizeref=axis_length / 10,
        #         anchor='tail',
        #         name=f'{colors[i]} axis'
        #     ))

        # # Layout
        # fig.update_layout(
        #     scene=dict(
        #         xaxis_title='X',
        #         yaxis_title='Y',
        #         zaxis_title='Z',
        #         aspectmode='data'
        #     ),
        #     title='3D Head Pose Visualization'
        # )

        # # Save as HTML
        # fig.write_html(output_file)
        # print(f"Figure saved to {output_file}")
        # ######################################################


        return headpose, verts
    
    def forward_flame(self, flame_param:dict[str, np.ndarray]):
        flame_param = {k: torch.from_numpy(v) for k, v in flame_param.items() if v.dtype == np.float32}

        flame_param = {
            'shape': flame_param['shape'][None],
            'static_offset': flame_param['static_offset'],
            'translation': flame_param['translation'],
            'rotation': flame_param['rotation'],
            'neck_pose': flame_param['neck_pose'],
            'jaw_pose': flame_param['jaw_pose'],
            'eyes_pose': flame_param['eyes_pose'],
            'expr': flame_param['expr'],
        }
        with torch.inference_mode():
            verts, _ = FLAME(
                flame_param['shape'],
                flame_param['expr'],
                flame_param['rotation'],
                flame_param['neck_pose'],
                flame_param['jaw_pose'],
                flame_param['eyes_pose'],
                flame_param['translation'],
                zero_centered_at_root_node=False,
                return_landmarks=False,
                return_verts_cano=True,
                static_offset=flame_param['static_offset'][:, :5023],
            )

        return verts.float().numpy()[0]

    def get_sequenceid(self, frame_id:str)-> str:
        return self.sequence_id
    
    def load_image(self, frame_id:str, camera_id: str)->np.ndarray:
        camera_idx = self.camera_id2idx[camera_id]
        path = self.dir / "images" / f"{int(frame_id):05d}_{camera_idx:02d}.png"
        img = self.read_image(path, apply_gamma_correction=False)
        return img


    def load_cleanplate(self, camera_id: str)->np.ndarray:  # TODO
        path = self.dir.parent/ f'BACKGROUND/image_{camera_id}.jpg'
        img = self.read_image(path=path, apply_gamma_correction=False)
        return img


    def load_sg(self, frame_id:str, camera_id: str)->np.ndarray:
        camera_idx = self.camera_id2idx[camera_id]
        path = self.dir / "segs" / f"{int(frame_id):05d}_{camera_idx:02d}.png"
        img = self.read_image(path, apply_gamma_correction=False)
        return img


    def load_fg_mask(self, frame_id:str, camera_id: str)->np.ndarray:
        camera_idx = self.camera_id2idx[camera_id]
        path = self.dir / "fg_masks" / f"{int(frame_id):05d}_{camera_idx:02d}.png"
        img = self.read_image(path)[..., :1]
        return img



class NersembleMultiCaptureDataset(base_dataset.BaseMultiCaptureDataset):
    """
    Dataset with CA2 assets for multiple captures

    recommended width=550, height=802,

    Args:
        captures: The unique identifiers of the mugsy captures in this dataset
        
    """
    SINGLE_CAPTURE_DATASET_CLS = NersembleSingleCaptureDataset


    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def filter_extra_camera_ids_by_capture(self, extra_camera_ids, capture):
        if extra_camera_ids is None:
            return None
        else:
            return_dict = dict()
            for k, v in extra_camera_ids.items():
                if k == 'ALL':  # key 'ALL' allows camera ids to be used in all captures
                    k = f'{capture.sid}--{capture.mcd}'
                elif (k.split('--')[0] == capture.sid) and (k.split('--')[1] == capture.mcd):
                    pass
                else:
                    continue
                return_dict[k] = v 
            return return_dict                    
        
    @staticmethod
    def folder_parser(base_dir: Path, training:bool|None = None) -> Tuple[List[NersembleCapture], List[Path]]:
        """
        Args:
            training: specifies split, if True: training split, if False: validation split, if None: all
        """
        captures = []
        dirs = []
        capture_paths = sorted(glob(f"{base_dir}/*/*"))
        capture_paths = [p for p in capture_paths if not p.endswith('BACKGROUND')]
        
        for capture_path in tqdm(capture_paths):
            capture = NersembleCapture(capture_path)
            if training is not None and training:
                if capture.sid in NERSEMBLE_VAL_IDS:
                    continue
            elif training is not None and not training:
                if capture.sid not in NERSEMBLE_VAL_IDS:
                    continue
            captures.append(capture)
            dirs.append(capture_path)
        
        return captures, dirs




class NersembleDataLoader(base_dataset.ParentDataLoader):
    DATASET_CLS = NersembleMultiCaptureDataset
