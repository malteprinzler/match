# This code has been adapted from https://github.com/facebookresearch/ava-256/blob/a9d2fbe85c2139d1c072212287b484309f3460e7/data/ava_dataset.py 
# Please refer to the license available under https://github.com/facebookresearch/ava-256/blob/a9d2fbe85c2139d1c072212287b484309f3460e7/LICENSE

import bisect
import io
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, TypeVar, Union
import einops
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm
import cv2
import io
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Union, List
import pudb
import cv2
from collections import defaultdict
from match.utils import general_util, file_util
from PIL import UnidentifiedImageError
from match.utils import geo_util
import einops
import numpy as np
from PIL import Image
from match.utils import data_util
from torchvision.transforms import InterpolationMode



T = TypeVar("T")

class MugsyCapture:
    """Unique identifier for a Mugsy capture"""

    def __init__(
        self,
        mcd: str,  # Mugsy capture date in 'yyyymmdd' format, eg `20210223`
        mct: str,  # Mugsy capture time in 'hhmm' format, eg `1023`
        sid: str,  # Subject ID, three letters and three numbers, eg `avw368`
    ):
        self.mcd = mcd
        self.mct = mct
        self.sid = sid

    def folder_name(self) -> str:
        return f"{self.mcd}--{self.mct}--{self.sid}"



class BaseSingleCaptureDataset(torch.utils.data.Dataset):
    """
    Dataset with Mugsy assets for a single capture


    Coordinate convention:
        image-space origin at top left, positive y direction is down, positive x: right, z: look-at
        top-left pixel center has coodinate 0.5, 0.5, bottom-right has w-0.5, h-0.5

        camera-space orientation: x: right, y: down, z: look-at
        
        world space orientation: (from perspective of subject) positive x: left, positive y: down, positive z: backwards

        
    """

    # conversion operations to bring different datasets to similar coordinate system
    # first rotates, then scales then subtracts scene center
    SCENE_ROTATION = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)  
    SCALE_FACTOR = np.array(1., dtype=np.float32) 
    SCENE_CENTER = np.array([0., 0., 0.], dtype=np.float32)

    PER_VIEW_KEYS = ['C2W', 'fxfycxcy', 'image', 'uv', 'sg_parts', 'mask', 'camera', 'bboxs']


    def __init__(
        self,
        capture: MugsyCapture,
        directory: str,
        height: int,
        width: int,
        camera_angles: List = None,
        nframes: int = -1,
        uv_directory: str = None,
        coarse_mesh_directory: str = None,
        frame_stride=1,
        head_crop = False,
        head_crop_height: int = None,
        head_crop_width: int = None,
        head_crop_offset_x: int=0,
        head_crop_offset_y: int=0,
        extra_camera_angles: list = None,
        verts_pad_to_n: int = None,  # padding vertices with zeros. Useful when combining samples from different sources into a batch
        skip_sequences = [],
        require_verts = True,
        require_segmentation = True,
    ):
        super().__init__()
        self.nframes = nframes
        self.capture = capture
        self.dir = Path(directory)
        self.height = height
        self.width = width
        self.camera_angles = camera_angles
        self.invalid_frame_cameras = defaultdict(list)  # for each frame, saves list of invalid cameras
        self.uv_directory = uv_directory
        self.coarse_mesh_directory = coarse_mesh_directory
        self.sid = self.capture.sid
        self.mcd = self.capture.mcd
        self.frame_stride = frame_stride
        self.head_crop = head_crop
        self.head_crop_height = head_crop_height
        self.head_crop_width = head_crop_width
        self.head_crop_offset_x = head_crop_offset_x
        self.head_crop_offset_y = head_crop_offset_y
        self.verts_pad_to_n = verts_pad_to_n
        self.extra_camera_angles = extra_camera_angles
        self.skip_sequences = skip_sequences
        self.require_verts = require_verts
        self.require_segmentation = require_segmentation


        # Tuple with all the identifiers of this capture, used in ddp-train
        self.identities = [capture]

        assert self.dir.exists(), f"Dataset directory {self.dir} does not seem to exist"

        # loading whatever metadata dataset needs for other preparation methods
        self.load_metadata()

        # Load krt dictionaries
        krt_dict = self.get_krt_dict()
        self.cameras = sorted(krt_dict.keys())

        # Pre-load krts in user-friendly dictionaries
        self.campos, self.camrot, self.intrin, self.distort = {}, {}, {}, {}
        for cam, krt in krt_dict.items():
            campos = (-np.dot(krt["extrin"][:3, :3].T, krt["extrin"][:3, 3])).astype(np.float32)
            campos = geo_util.apply_rotation_scale_center_to_points(campos[None], rotation = self.SCENE_ROTATION, scale=self.SCALE_FACTOR, center=self.SCENE_CENTER)[0]
            self.campos[cam] = campos
            
            camrot = (krt["extrin"][:3, :3]).astype(np.float32)
            camrot = camrot @ self.SCENE_ROTATION.T  # world to camera rotation
            self.camrot[cam] = camrot

            self.intrin[cam] = krt["intrin"].astype(np.float32)
            self.distort[cam] = krt['dist'].astype(np.float32)

        self.camera_map = dict()
        for i, cam in enumerate(self.cameras):
            self.camera_map[cam] = i

    def load_metadata(self):
        """loads any metadata that may be used for later computations."""
        pass

    @property
    def height_original(self)->int:
        raise NotImplementedError
    
    @property
    def width_original(self)->int:
        raise NotImplementedError


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
        raise NotImplementedError

    def load_image(self, frame_id:str, camera_id: str)->np.ndarray:
        raise NotImplementedError

    def load_sg(self, frame_id:str, camera_id: str)->np.ndarray:
        raise NotImplementedError

    def load_fg_mask(self, frame_id:str, camera_id: str)->np.ndarray:
        raise NotImplementedError
    
    def load_coarse_verts(self, frame_id:str) -> np.ndarray:
        verts = np.zeros((0, 3), dtype=np.float32)
        if self.coarse_mesh_directory is not None:
            seq_id = self.get_sequenceid(frame_id)
            path = file_util.Path(self.coarse_mesh_directory, self.sid, seq_id, frame_id, 'verts.npy')
            if path.exists():
                verts = np.load(path).astype(np.float32)
            else:
                raise data_util.SkippableError(f'Couldnt find coarse mesh prediction under path {path}.')
        return verts
        
    def load_uv(self, frame_id:str, camera_id:str)->np.ndarray:
        if self.uv_directory is None:
            return None
        else:
            seq_id = self.get_sequenceid(frame_id)
            filename = f'uv_cam{camera_id}.png'
            path = file_util.Path(self.uv_directory, self.sid, seq_id, frame_id, filename)
            img_bytes = path.read_bytes()
            with io.BytesIO(img_bytes) as b:
                with Image.open(b) as img:
                    img = np.asarray(img) 
            assert img.dtype == np.uint8
            return img

    def check_image(self, frame_id:str, camera_id=str)->bool:
        raise NotImplementedError

    def check_data_on_disk(self, frame_id:str, camera_ids: List[str]):
        for camera_id in camera_ids:
            self.check_image(frame_id=frame_id, camera_id=camera_id)

    def get_vertex_groups(self) -> dict[str, list[int]]:
        raise NotImplementedError
    
    def fetch_data_from_disk(self, frame_id: str, camera_ids: List[str], extra_camera_ids: List[str] = []) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        if any([camera_id in self.invalid_frame_cameras[frame_id] for camera_id in camera_ids]):
            raise data_util.SkippableError(f'Skipping known invalid frame-camera combination. Requested camera ids: {camera_ids} for frame {frame_id} in {self.dir} but known invalid camera ids are {self.invalid_frame_cameras[frame_id]}.')
        
        try:
            # Head pose (global transform of the person's head)
            headpose, verts = self.load_headpose_and_verts(frame_id)        
            verts = geo_util.apply_rotation_scale_center_to_points(verts, rotation=self.SCENE_ROTATION, scale=self.SCALE_FACTOR, center=self.SCENE_CENTER)
            headpose[:3,:3] = self.SCENE_ROTATION @ headpose[:3, :3]
            headpose[:3, -1] = geo_util.apply_rotation_scale_center_to_points(headpose[:3, -1][None], rotation=self.SCENE_ROTATION, scale=self.SCALE_FACTOR, center=self.SCENE_CENTER)[0]
            coarse_verts = self.load_coarse_verts(frame_id=frame_id)
            coarse_verts = geo_util.apply_rotation_scale_center_to_points(coarse_verts, rotation=self.SCENE_ROTATION, scale=self.SCALE_FACTOR, center=self.SCENE_CENTER)

        except FileNotFoundError as e:
            self.invalid_frame_cameras[frame_id].append('ALL')
            raise data_util.SkippableError(f'Error loading frame data for {self.dir} frame {frame_id}:\n{e}')

        # sequence
        sequence_id = self.get_sequenceid(frame_id)
        
        all_camera_ids = list(set(camera_ids) | set(extra_camera_ids))

        # per camera features
        imgs = list()
        uvs = list()
        sg_parts = list()
        fg_masks = list()
        Rts = list()
        Ks = list()
        bboxs = list()
        for camera_id in all_camera_ids:
            # Prepare rotation and translation matrices
            # sample['camrot']: rotation matrix R (world→camera rotation)
            # sample['campos']: camera center in world coordinates (C)
            # World-to-camera RT has form [R | t] with t = -R @ C
            camrot = self.camrot[camera_id]
            campos = self.campos[camera_id]
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = camrot
            Rt[:3, 3] = -camrot.dot(campos)
            Rts.append(Rt)

            try:
                # load raw images
                img = self.load_image(frame_id=frame_id, camera_id=camera_id)
                try:
                    uv = self.load_uv(frame_id=frame_id, camera_id=camera_id)
                except Exception as e:
                    # if not camera_id in extra_camera_ids:  # don't need the uvmaps for extra cameras
                    #     print(f'WARNING: DIDNT FIND UVMAP OF INPUT IMAGE: {self.capture.sid}-{frame_id}-{camera_id}')
                    uv = np.zeros_like(img)

                sg = self.load_sg(frame_id=frame_id, camera_id=camera_id)
                if sg is None:
                    if self.require_segmentation:
                        raise FileNotFoundError(f'Couldnt find segmentation parts for {self.capture.sid}-{frame_id}-{camera_id}.')
                    sg = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                fg_mask = self.load_fg_mask(frame_id=frame_id, camera_id=camera_id)
                if uv is None:
                    uv = np.zeros_like(img)

                # resize images
                K = self.intrin[camera_id].copy()
                K[0] *= self.width / self.width_original
                K[1] *= self.height / self.height_original
                img, uv, sg, fg_mask = [cv2.resize(x, (self.width, self.height), interpolation=inter) 
                    for x, inter in [(img, cv2.INTER_LINEAR), (uv, cv2.INTER_NEAREST), (sg, cv2.INTER_NEAREST), (fg_mask, cv2.INTER_NEAREST)]]

                image_likes = dict(img=img, uv=uv, sg=sg, fg_mask=fg_mask)
                
                # 0...255 -> 0...1, HWC->CHW
                for k, v in image_likes.items():
                    if k not in ['sg']:
                        v = v.astype(np.float32)/255
                    if len(v.shape) == 2: 
                        v = v[..., None]
                    v = einops.rearrange(v, 'H W C -> C H W')
                    image_likes[k] = v
                
                # crop around head                
                if self.head_crop:
                    C, H, W = image_likes['img'].shape
                    crop_w_in = crop_w_out = self.head_crop_width
                    crop_h_in = crop_h_out = self.head_crop_height
                    head_center_world = headpose[:, -1]  # (4,)

                    head_center_cam = einops.einsum(Rt, head_center_world, 'i j, j -> i')[:3]  # (3,)
                    head_center_screen = einops.einsum(K, head_center_cam, 'i j, j -> i')  # (3,)
                    head_center_screen[:2] = head_center_screen[:2] / head_center_screen[2:]
                    crop_center = head_center_screen[:2]
                    crop_center[0] += self.head_crop_offset_x
                    crop_center[1] += self.head_crop_offset_y
                    crop_t = crop_center[1] - crop_h_in / 2
                    crop_l = crop_center[0] - crop_w_in / 2

                    # making sure entire head crop is on image, if not, move such that it is
                    crop_t = np.maximum(np.minimum(crop_t, np.ones_like(crop_t) * (H-crop_h_in)), np.zeros_like(crop_t))
                    crop_l = np.maximum(np.minimum(crop_l, np.ones_like(crop_l) * (W-crop_w_in)), np.zeros_like(crop_l))
                    crop_t = crop_t.astype(np.int32)   
                    crop_l = crop_l.astype(np.int32)  
                    if np.any(crop_l + crop_w_in > W):
                        crop_h_in = int(np.min(crop_h_in * (W-crop_l) / crop_w_in))
                        crop_w_in = int(np.min(W-crop_l))
                    if np.any(crop_t + crop_h_in > H):
                        crop_w_in = int(np.min(crop_w_in * (H-crop_t) / crop_h_in))
                        crop_h_in = int(np.min(H-crop_t))

                    # cropping
                    for k, v in image_likes.items():
                        v = v[:, crop_t: crop_t+crop_h_in, crop_l:crop_l + crop_w_in]
                        image_likes[k] = v
                    K[0, 2] = K[0, 2] - crop_l
                    K[1, 2] = K[1, 2] - crop_t

                    # resizing to target output resolution
                    if (crop_w_in != crop_w_out) or (crop_h_in != crop_h_out):
                        for k, v in image_likes.items():
                            if k in ['sg', 'uv']:
                                interpolation_mode = InterpolationMode.NEAREST 
                            elif k in ["img", "bg", "fg_mask"]:
                                interpolation_mode = InterpolationMode.BILINEAR
                            else:
                                raise KeyError(f'unhandled key {k}')
                            v = TF.resize(v, (crop_h_out, crop_w_out), interpolation=interpolation_mode)
                            image_likes[k] = v
                        K[0] =  K[0] * crop_w_out / crop_w_in
                        K[1] =  K[1] * crop_h_out / crop_h_in
                    bboxs.append([crop_l, crop_t, crop_l + crop_w_in, crop_t+crop_h_in])
                else:
                    c, h, w = image_likes['img'].shape
                    bboxs.append([0, 0, w, h])

                Ks.append(K)
                imgs.append(image_likes["img"])
                uvs.append(image_likes["uv"])
                sg_parts.append(image_likes["sg"])
                fg_masks.append(image_likes["fg_mask"])

            except (FileNotFoundError, UnidentifiedImageError) as e:
                self.invalid_frame_cameras[frame_id].append(camera_id)
                raise data_util.SkippableError(f'Error loading samples for {self.dir}, frame: {frame_id}, camera: {camera_id}\n{e}')

        imgs = np.stack(imgs)
        sg_parts = np.stack(sg_parts)
        fg_masks = np.stack(fg_masks)
        uvs = np.stack(uvs)
        Ks = np.stack(Ks)
        Rts = np.stack(Rts)
        all_camera_ids_np = np.array(list(map(int, all_camera_ids)))
        bboxs = np.array(bboxs)
        frame_id = frame_id
        subject_id = self.capture.sid

        V, C, H, W = imgs.shape

        # padding vertices if necessary
        if self.verts_pad_to_n is not None:
            verts_ = np.zeros((self.verts_pad_to_n, 3), dtype=verts.dtype)
            verts_[:len(verts)] = verts
            verts = verts_        

        C2W = geo_util.invert_c2w(Rts)
        fxfycxcy = np.stack([
            Ks[:, 0, 0] / W, 
            Ks[:, 1, 1] / H,
            Ks[:, 0, 2] / W,
            Ks[:, 1, 2] / H,
            ], 
            axis=-1)

        sample=dict(
            verts=verts,
            coarse_verts=coarse_verts,
            C2W=C2W,
            fxfycxcy=fxfycxcy,
            image=imgs,
            uv=uvs,
            sg_parts=sg_parts,  # sapiens segmentations
            mask=fg_masks,
            headpose=headpose,
            frame=frame_id,
            camera=all_camera_ids_np,
            subject=subject_id,
            sequence = sequence_id,
            scene_rotation = self.SCENE_ROTATION,
            scene_center = self.SCENE_CENTER,
            scene_scale = self.SCALE_FACTOR,
            bboxs = bboxs,
        )

        # cast all numpy arrays to torch tensors
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
                sample[k] = v

        # splitting camera_ids and extra_camera_ids again
        camera_id_idcs = np.array([all_camera_ids.index(i) for i in camera_ids])
        extra_camera_id_idcs = np.array([all_camera_ids.index(i) for i in extra_camera_ids])
        for k in self.PER_VIEW_KEYS:
            if extra_camera_ids:
                sample['extra_'+k] = sample[k][extra_camera_id_idcs]
            sample[k] = sample[k][camera_id_idcs]
        return sample

    def get_sequenceid(self, frame_id:str)->str:
        raise NotImplementedError

    def sample_cameras_from_camera_angles(self, frame_id:str, camera_angles=None, avoid_camera_ids=None):
        valid_cam_ids = [k for k in self.cameras if k not in self.invalid_frame_cameras[frame_id]]
        if avoid_camera_ids is not None:
            valid_cam_ids = [k for k in valid_cam_ids if not k in avoid_camera_ids]
        if camera_angles is None:
            camera_angles = self.camera_angles
        if len(valid_cam_ids) == 0:
            self.invalid_frame_cameras[frame_id].append('ALL')
            raise data_util.SkippableError(f'No valid cameras found for {self.dir} frame: {frame_id}')
        if camera_angles is None:
            return valid_cam_ids
        else:            
            campos = np.stack([self.campos[k] for k in valid_cam_ids])
            if len(campos) < len(camera_angles):
                raise data_util.SkippableError(f'Not enough valid cameras found for {self.dir} frame: {frame_id}')
            cam_idcs = data_util.sample_cameras_from_angles(campos, camera_angles, 
                                                             temperature=1,
                                                             ncams_per_angle=1)
            return [valid_cam_ids[i] for i in cam_idcs]
        
    def get_extra_camera_ids(self, frame_id:str, avoid_camera_ids=[]):
        '''
        Args:
            avoid_camera_ids only avoids the ids if sampling the extra cameras from an angle range, not if they are explicitly specified through self.extra_camera_ids
        '''
        if self.extra_camera_angles is not None:
            return self.sample_cameras_from_camera_angles(camera_angles=self.extra_camera_angles, avoid_camera_ids=avoid_camera_ids, frame_id=frame_id)
        else:
            return []
        
    def load_headpose_and_verts(frame_id: str)-> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @staticmethod
    def validate_camera(base_dir: str, camera_id: str)->bool:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Optional[Dict[str, Union[np.ndarray, int, str]]]:
        exception_counter = 0
        while True:
            try:
                frame_id = self.get_frameid((idx+exception_counter) % len(self))
                if 'ALL' in self.invalid_frame_cameras[frame_id]:
                    raise data_util.SkippableError(f'Skipping known invalid {self.dir} frame: {frame_id}')
                camera_ids = self.sample_cameras_from_camera_angles(frame_id)
                extra_camera_ids = self.get_extra_camera_ids(frame_id, avoid_camera_ids = camera_ids)
                return self.fetch_data_from_disk(frame_id=frame_id, camera_ids=camera_ids, extra_camera_ids=extra_camera_ids)
            except data_util.SkippableError as e:
                logging.warning(f'Skippable error in loading sample {self.dir} {frame_id} {idx+exception_counter} ({exception_counter} Retries):\n{e}\nRetrying next sample ...')
                exception_counter += 1                
    
    def get_image_path(self, frame_id:str, camera_id:str):
        raise NotImplementedError


    def get_frameid(self, idx:int) -> str:
        raise NotImplemented

    def __len__(self):
        raise NotImplementedError

    ### Methods added for compat with previous version of dataset. Might want to revisit these
    def get_allcameras(self) -> Set[str]:
        return set(self.cameras)



class BaseMultiCaptureDataset(torch.utils.data.Dataset):
    """
    Combines several SingleCaptureDatasets into one Dataset
    """

    SINGLE_CAPTURE_DATASET_CLS = BaseSingleCaptureDataset

    def __init__(
        self,
        root_path:str,
        height: int,
        width: int,
        max_captures: int|None = None,
        camera_angles: List[str] = None,
        training: bool|None =None,
        deterministic_shuffle: bool = False,
        process_idx = 0,
        world_size=1,
        coarse_mesh_directory:str=None,
        uv_directory:str=None,
        frame_stride = 1,
        exclude_subjects =list(),
        only_subjects = list(),
        invalid_captures_path:str = None, 
        head_crop = False,
        head_crop_height: int = None,
        head_crop_width: int = None,
        head_crop_offset_y: int = 0,
        head_crop_offset_x: int = 0,
        extra_camera_angles = None,
        skip_sequences = list(),
        nsamples = None,
        **kwargs,
    ):
        super().__init__()

        captures, dirs = self.folder_parser(root_path, training=training)
        captures, dirs = self.filter_captures(root_path, 
            captures, 
            dirs, 
            exclude_subjects=exclude_subjects, 
            only_subjects=only_subjects, 
            invalid_captures_path=invalid_captures_path, 
            process_idx=process_idx, 
            world_size=world_size, 
            max_captures=max_captures,
            skip_sequences=skip_sequences
        )
        self.captures = captures
        self.dirs = dirs
        self.height = height
        self.width = width
        self.process_idx = process_idx
        self.world_size = world_size
        self.deterministic_shuffle = deterministic_shuffle
        self.extra_camera_angles = extra_camera_angles
        self.nsamples = nsamples


        # Tuples with all the identifiers of this capture, used in ddp-train
        self.identities = captures

        # Load the single-capture datasets
        self.single_capture_datasets = OrderedDict()
        for capture, capture_dir in tqdm(
            zip(captures, dirs),
            desc="Loading single id captures",
            total=len(captures),
        ):
            self.single_capture_datasets[capture] = self.SINGLE_CAPTURE_DATASET_CLS(
                capture=capture, directory=capture_dir, height=height, width=width, camera_angles=camera_angles,
                head_crop=head_crop, head_crop_height=head_crop_height, head_crop_width=head_crop_width,
                head_crop_offset_x=head_crop_offset_x, 
                head_crop_offset_y=head_crop_offset_y, 
                coarse_mesh_directory=coarse_mesh_directory, uv_directory=uv_directory, frame_stride=frame_stride,
                extra_camera_angles = self.extra_camera_angles,
                skip_sequences=skip_sequences,
                **kwargs,
            )
        
        self.post_single_capture_dataset_loading_hook()


    def post_single_capture_dataset_loading_hook(self):
        # Dataset lengths
        self.cumulative_sizes = np.cumsum([len(x) for x in self.single_capture_datasets.values()])
        self.total_len = self.cumulative_sizes[-1]

        # Deterministic shuffling if specified
        self.sample_idcs = np.arange(self.total_len)
        if self.deterministic_shuffle:
            rng = np.random.default_rng(0)
            self.sample_idcs = rng.permutation(self.sample_idcs)
        if self.nsamples is not None:
            self.total_len = min(self.total_len, self.nsamples)
    
    @staticmethod
    def filter_captures(root, captures, dirs, exclude_subjects, only_subjects, invalid_captures_path:str=None, max_captures=None, process_idx=0, world_size=1, equal_captures_per_subject=False, skip_sequences=[]):
        
        # filtering using file under invalid_captures_path
        if invalid_captures_path is not None:
            with open(invalid_captures_path, 'r') as f:
                invalid_dirs = [l.strip().strip('/') for l in f.readlines()]
            captures_, dirs_ = list(), list()
            for c, d in zip(captures, dirs):
                if not str(d).replace(root, '').strip('/') in invalid_dirs:
                    captures_.append(c)
                    dirs_.append(d)
            captures = captures_
            dirs = dirs_            
        
        # filtering by subject
        filtered_captures = list()
        filtered_dirs = list()
        for c, d in zip(captures, dirs):
            if c.sid in exclude_subjects:
                continue
            if len(only_subjects)>0 and c.sid not in only_subjects:
                continue
            filtered_captures.append(c)
            filtered_dirs.append(d)
        captures = filtered_captures
        dirs = filtered_dirs

        # filtering by sequence
        if skip_sequences:
            filtered_captures = list()
            filtered_dirs = list()
            for c, d in zip(captures, dirs):
                if any([c.mcd.startswith(s) for s in skip_sequences]):
                    continue
                filtered_captures.append(c)
                filtered_dirs.append(d)
            captures = filtered_captures
            dirs = filtered_dirs

        if equal_captures_per_subject:
            subject_capture_idcs = defaultdict(list)
            subject_equalized_capture_idcs = list()
            for i, c in enumerate(captures):
                subject_capture_idcs[c.sid].append(i)
            
            while len(subject_capture_idcs.keys()):
                subjects = sorted(subject_capture_idcs.keys())
                for s in subjects:
                    subject_equalized_capture_idcs.append(subject_capture_idcs[s].pop(0))
                    if len(subject_capture_idcs[s])==0:
                        subject_capture_idcs.pop(s)  
            captures = [captures[i] for i in subject_equalized_capture_idcs]               
            dirs = [dirs[i] for i in subject_equalized_capture_idcs]       

        if (max_captures is not None) and (max_captures > 0):
            captures = captures[:max_captures]
            dirs = dirs[:max_captures]

        captures = general_util.split_into_chunks(captures, world_size)[process_idx]
        dirs = general_util.split_into_chunks(dirs, world_size)[process_idx]
        return captures, dirs

    @staticmethod
    def folder_parser(base_dir: Path, max_captures: int = None, training:bool|None = None) -> Tuple[List[MugsyCapture], List[Path]]:
        raise NotImplementedError


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Inspired from PyTorch's ConcatDataset"""

        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        
        idx = self.sample_idcs[idx]  # used for potentially deterministically shuffling data

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        capture = self.captures[dataset_idx]
        sample = self.single_capture_datasets[capture][sample_idx]

        if sample is not None:
            sample["idindex"] = dataset_idx

        return sample
    
    def __len__(self):
        return self.total_len

    def get_allcameras(self) -> Set[str]:
        """Get all the cameras in this dataset"""
        other_cameras = [x.get_allcameras() for x in self.single_capture_datasets.values()]
        return set().union(*other_cameras)


class ParentDataLoader(torch.utils.data.DataLoader):
    DATASET_CLS = None  
    def __init__(self, dataloader_kwargs, dataset_kwargs:dict, training:bool|None=None):
        # setting kwarg defaults for training / validation dataloaders and datasets
        dataset_kwargs['training'] = dataset_kwargs.get('training', training)
        if training is not None and training:
            dataloader_kwargs['shuffle'] = dataloader_kwargs.get('shuffle', True)
        elif not training:  # validation
            dataloader_kwargs['shuffle'] = dataloader_kwargs.get('shuffle', False)

        dataset = self.DATASET_CLS(**dataset_kwargs)
        super().__init__(dataset=dataset, **dataloader_kwargs)
