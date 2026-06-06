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


class TrackedDataset(torch.utils.data.Dataset):
    def __init__(self, dir_list, cameras=[], frame_stride=1, shape=None, mean=None, std=None, pad=False):
        self.tracking_dirs = sorted(dir_list)
        if shape:
            assert len(shape) == 2
        if mean or std:
            assert len(mean) == 3
            assert len(std) == 3
        self.shape = shape
        self.mean = torch.tensor(mean) if mean else None
        self.std = torch.tensor(std) if std else None
        self.pad = pad

        self.image_list = list()  # saves paths to images
        for p in self.tracking_dirs:
            p = Path(p)
            if p.name.startswith('UNION'): 
                continue
            imgdir = p/'images'
            image_list_ = sorted([p for p in imgdir.iterdir() if p.suffix.lower() in ['.png', '.jpg']])
            image_frame_ids_ = [int(p.name.split('_')[0]) for p in image_list_]
            image_cam_ids_ = [int(p.name.split('_')[1].split('.')[0]) for p in image_list_]
            image_list_ = [str(p) for p, fid in zip(image_list_, image_frame_ids_) if fid % frame_stride == 0]
            image_cam_ids_ = [cid for cid, fid in zip(image_cam_ids_, image_frame_ids_) if fid % frame_stride == 0]
            if cameras:
                image_list_ = [p for p, cid in zip (image_list_, image_cam_ids_) if cid in cameras]
            self.image_list.extend(image_list_)            


    def __len__(self):
        return len(self.image_list)
    
    def _preprocess(self, img):
        if self.shape:
            if self.pad:  # black padding
                print('PADDING')
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
        img_path = self.image_list[idx]
        with Image.open(img_path) as img:
            orig_img = np.asarray(img) 
        assert orig_img.dtype == np.uint8

        orig_img = gamma_correction_cv(orig_img, gamma=2.2)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR) 
        img = self._preprocess(orig_img)
        
        ret_path = img_path
        return ret_path, orig_img, img
    

def gamma_correction_cv(img: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")
    return cv2.LUT(img, table)