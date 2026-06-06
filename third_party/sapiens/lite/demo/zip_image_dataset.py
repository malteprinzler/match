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