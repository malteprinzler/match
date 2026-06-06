import pudb
import torch
import einops
from tqdm import tqdm
from torchvision.transforms import Resize, InterpolationMode
from match.utils import vis_util


class SapiensEncoder:
    def __init__(self, ckpt, device, patch_size:int, dtype=torch.bfloat16, prediction_tiles_sqrt=None):
        self.dtype = dtype
        self.device = device
        self.patch_size = patch_size
        self.ckpt = ckpt
        self.mean = torch.tensor([123.5, 116.5, 103.5], dtype=self.dtype, device=self.device)/255
        self.std = torch.tensor([58.5, 57.0, 57.5], dtype=self.dtype, device=self.device) / 255
        self.prediction_tiles_sqrt = prediction_tiles_sqrt  # puts images into tiles before running Sapiens to save time. defines edge length of tile grid so n_tiles = prediction_tiles_sqrt**2
        self.resize = Resize((1024, 1024), interpolation=InterpolationMode.BILINEAR)
        self.model = None
        self.C = self.get_outchannels()
        self.model = self.init_model()

    def init_model(self):
        model = torch.jit.load(self.ckpt)
        model.to(dtype=self.dtype, device=self.device)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
        return model

    def get_outchannels(self):
        model_version = Path(self.ckpt).name.split('_')[1]
        outchannel_dict = {'1b': 1536}
        return outchannel_dict[model_version]

    @torch.no_grad()
    def __call__(self, imgs):
        '''
        
        Args:
            imgs: images of shape (B, 3, H, W), 0 ... 1

        Returns:
            features: (B, C, H_, W_)  patchified features (H_ = H/self.patch_size, W_=W/self.patch_size)
        '''

        imgs = imgs.to(dtype=self.dtype, device=self.device)
        sample_locations = self.get_sample_locations(imgs)
        imgs = (imgs - einops.rearrange(self.mean, 'c -> 1 c 1 1'))/einops.rearrange(self.std, 'c -> 1 c 1 1')
        imgs, sample_locations = self.to_grid(imgs, sample_locations)  
        imgs = self.resize(imgs)
        features = self.model(imgs)[0]  # (B, C, H, W)
        features = self.sample_features(features, sample_locations)
        features = self.ungrid_features(features)

        return features.detach()
    
    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        changed = False
        if dtype is not None and dtype != self.dtype:
            changed = True
            self.dtype = dtype
        if device is not None and device != self.dtype:
            changed = True
            self.device=device

        if changed:
            self.mean = self.mean.to(device=self.device, dtype=self.dtype)
            self.std = self.std.to(device=self.device, dtype=self.dtype)
            self.model = self.init_model()
        return self
            
    
    def get_sample_locations(self, imgs:torch.Tensor):
        '''
        
        Returns:
            sample_locations (B, n_v, n_h, 2) normalized to 0 ... 1; (0,0)=top-left corner of top-left pixel, (1,1)=bottom-right corner of bottom-right pixel
        '''
        B, C, H, W = imgs.shape
        device=imgs.device
        dtype=imgs.dtype
        assert H % self.patch_size ==0
        assert W % self.patch_size ==0
        patch_centers = torch.stack(torch.meshgrid(torch.arange(self.patch_size/2, W, self.patch_size, device=device, dtype=dtype), 
                                                   torch.arange(self.patch_size/2, H, self.patch_size, device=device, dtype=dtype), indexing='xy'),dim=-1)

        # 0...L -> 0 ... 1
        patch_centers[..., 0] = patch_centers[..., 0] / W
        patch_centers[..., 1] = patch_centers[..., 1] / H 
        patch_centers = torch.repeat_interleave(patch_centers[None], B, dim=0)

        return patch_centers


    
    def to_grid(self, imgs, sample_locations):
        ''' Arranges images in grid to make inference faster
        Args:
            imgs: (B, C, H, W)
            sample_locations: (B, h, w, 2) (0...1)
        '''
        if self.prediction_tiles_sqrt is not None:
            device = imgs.device
            dtype=imgs.dtype
            imgs = einops.rearrange(imgs, '(b n_v n_h) c h w -> b c (n_v h) (n_h w)', n_v=self.prediction_tiles_sqrt, n_h=self.prediction_tiles_sqrt)
            sample_locations = einops.rearrange(sample_locations, '(b n_v n_h) h w c -> b n_v n_h h w c', n_v=self.prediction_tiles_sqrt, n_h=self.prediction_tiles_sqrt)
            sample_locations = sample_locations  / self.prediction_tiles_sqrt
            sample_locations[..., 0] = sample_locations[..., 0] + einops.rearrange(torch.arange(0, 1, 1/self.prediction_tiles_sqrt, device=device, dtype=dtype), 'n_h -> 1 1 n_h 1 1')
            sample_locations[..., 1] = sample_locations[..., 1] + einops.rearrange(torch.arange(0, 1, 1/self.prediction_tiles_sqrt, device=device, dtype=dtype), 'n_v -> 1 n_v 1 1 1')
            sample_locations = einops.rearrange(sample_locations, 'b n_v n_h h w c -> b (n_v h) (n_h w) c')


        return imgs, sample_locations 

    def ungrid_features(self, features):
        '''
        Args:
            features (B, C, h, w)
        '''
        if self.prediction_tiles_sqrt is not None:
            features = einops.rearrange(features, 'b c (n_v h) (n_h w) -> (b n_v n_h) c h w', n_v = self.prediction_tiles_sqrt, n_h=self.prediction_tiles_sqrt)
        return features

    
    def sample_features(self, features, sample_locations):
        '''
        
        Args:
            features (B, C, H, W)
            sample_locations: (B, h, w, 2) normalized from 0...1
        '''
        sample_locations = sample_locations * 2 - 1  # [0...1]-> [-1 ... 1]
        features = torch.nn.functional.grid_sample(features, sample_locations, mode='nearest', padding_mode='border', align_corners=False)
        return features


    
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import cv2
from zipfile import ZipFile
from zipp import Path as ZipPath
from PIL import Image
import numpy as np
import io
import os
from pathlib import Path
import pandas as pd
import pillow_avif



class MultiZipImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_list, cameras=[], frame_stride=1, shape=None, mean=None, std=None, pad=False):
        self.zip_list = sorted(image_list)
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        self.pad = pad

        self.zip_image_list = list()  # saves tuples of (zipfilepath, image_path within zipfile)
        for p in self.zip_list:
            if p.endswith('.zip'):
                cam_name = os.path.basename(p).strip('.zip').strip('cam')
                if (len(cameras)>0) and (cam_name not in cameras):
                    continue
                frame_file = Path(p).parents[1] / 'frame_list.csv'
                framelist = pd.read_csv(str(frame_file), dtype=str, sep=r",")
                framelist = framelist['frame_id'].iloc[::frame_stride]
                
                image_list_ = sorted(ZipFile(p).namelist())
                for frame in framelist:
                    img_name = f'cam{cam_name}/{int(frame):06d}.avif'
                    if img_name in image_list_:
                        self.zip_image_list.append((p, img_name))
            elif '.zip/' in p:
                assert frame_stride == 1, 'Input image list given with specified images already. Expected frame stride to be 1 to satisfy frame stride of given list'
                zip_path, content_name = p.split('.zip/')
                zip_path = zip_path + '.zip'
                self.zip_image_list.append((zip_path, content_name))
            
            else:
                raise NotImplementedError(f'Not implemented path type {p}')


    def __len__(self):
        return len(self.zip_image_list)
    
    def _preprocess(self, img):
        if self.shape:
            if self.pad:  # black padding
                resize_factor = min(self.shape[0] / img.shape[0], self.shape[1] / img.shape[1])
                img = cv2.resize(img, (int(round(img.shape[1]*resize_factor)), int(round(img.shape[0]*resize_factor))), interpolation=cv2.INTER_LINEAR)
                img_ = np.zeros((self.shape[0], self.shape[1], 3), dtype=np.uint8)
                pad_l = int(((self.shape[1] - img.shape[1]) / 2 // 16) * 16)
                pad_t = int(((self.shape[0] - img.shape[0]) / 2 // 16) * 16)
                img_[pad_t:pad_t + img.shape[0], pad_l: pad_l + img.shape[1]] = img
                img = img_
            else:
                img = cv2.resize(img, (self.shape[1], self.shape[0]), interpolation=cv2.INTER_LINEAR)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img)
        img = img[[2, 1, 0], ...].float()
        if self.mean is not None and self.std is not None:
            mean=self.mean.view(-1, 1, 1)
            std=self.std.view(-1, 1, 1)
            img = (img - mean) / std
        return img
    
    def __getitem__(self, idx):
        zippath, img_path = self.zip_image_list[idx]
        with ZipFile(zippath) as f:
            orig_img_bytes = ZipPath(f, img_path).read_bytes()
        with io.BytesIO(orig_img_bytes) as b: 
            with Image.open(b) as img:
                orig_img = np.asarray(img) 
        assert orig_img.dtype == np.uint8

        orig_img = gamma_correction_cv(orig_img, gamma=2.2)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
        img = self._preprocess(orig_img)
        
        ret_path = f'{zippath}/{img_path}'
        return ret_path, orig_img, img
    

def gamma_correction_cv(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)
    

if __name__ == '__main__':
    import time
    ckpt = '/fast/mprinzler/sapiens/sapiens_lite_host/torchscript/pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_torchscript.pt2'
    device = torch.device('cuda')
    dtype=torch.bfloat16
    from match.data.ava256_dataset import AvaMultiCaptureDataset
    import matplotlib.pyplot as plt
    patch_size = 8
    encoder = SapiensEncoder(ckpt=ckpt, device=device, dtype=dtype, patch_size=patch_size, prediction_tiles_sqrt=2)
    root = '/fast/mprinzler/gintern/datasets/ava-256'
    cam_angles = [[0,0], [0, -15],
                  [20,0], [20, 36], [20, -15],
                  [-20,0], [-20, 36], [-20, -15],
                  [40,-5], [40, 25],
                  [-40,-5], [-40, 25],
                  ]
    ds = AvaMultiCaptureDataset(root_path=root, 
                                max_captures=10, 
                                # cameras_specified=cam_ids, 
                                # frames_per_subject=1, 
                                # deterministic_shuffle=True,
                                stage1_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/gtempeh/easyava256_tempehuv_uvres512-240k',
                                sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
                                # sapiens_feature_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_features/framestride_100_fixedresizing/sapiens_2b',
                                # sapiens_feature_patchsize = 8,
                                uv_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/gtempeh/easyava256_tempehuv_uvres512-240k',
                                camera_angles_specified=cam_angles, 
                                equal_captures_per_subject = True,                                
                                deterministic_shuffle=True,
                                training=None,
                                frame_stride = 10,
                                # exclude_subjects = ['FXN596'],
                                height=786, width=512,
                                head_crop=True,
                                head_crop_height=512+128,
                                head_crop_width=512,
                                head_crop_offset_y = -64,
                                # random_origin_std = 0.15,
                                # random_rotation_angle = 30.,
                                # brightness_range = 0.2,
                                # contrast_range = 0.2,
                                # saturation_range = 0.2,
                                # hue_range = 0.1,
                                # p_grayscale = 0.2,
                                # camera_sampling_temperature = 4,
                                )
    sample = ds[0]
    nbatches = 100
    for i in tqdm(range(nbatches+1)):
        if i==1:
            t_start = time.time()
        features = encoder(sample['image'])
        if i==nbatches: 
            t_end = time.time()
            break       



