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


import os
from pathlib import Path
import sys
import torch
import torch.utils.data

from loguru import logger
from omegaconf import OmegaConf
from loguru import logger
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.trainer import RegressorTrainer, RegressorMode
from utils.general import build_dataset, build_loader, seed_everything, to_device
torch.backends.cudnn.benchmark = True
import pudb
from torch.utils.data._utils.collate import default_collate
import copy
from data.base import DatasetMode
import numpy as np
import tqdm


def train(config):

    val_config = copy.deepcopy(config)
    val_config.data.join_configs = False
    val_dataset = build_dataset(val_config, camera_list=[val_config.data.test_camera], mode=DatasetMode.validation)
    val_loader = build_loader(val_dataset, **{**config.train, **config.data})

    test_config = copy.deepcopy(config)
    test_config.data.join_configs = False
    test_dataset = build_dataset(test_config, camera_list=[test_config.data.test_camera], mode=DatasetMode.test)
    test_loader = build_loader(test_dataset, **{**config.train, **config.data})

    cross_config = copy.deepcopy(config)
    cross_config.data.join_configs = True
    source_config = OmegaConf.load(cross_config.data_cross_config)
    cross_dataset = build_dataset(cross_config, camera_list=[cross_config.data.test_camera], source_config=source_config, mode=DatasetMode.validation)
    cross_loader = build_loader(cross_dataset, **{**config.train, **config.data})


    trainer = RegressorTrainer(config, val_dataset)
    trainer.restore(force=True)

    trainer.model.bg_color = "white"
    trainer.eval()
    for loader_name, loader in [('val', val_loader), ('test', test_loader), ('cross', cross_loader)]:
        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'{loader_name} Dataset')):
            assert len(batch['exp']) == 1
            if trainer.current_sentence != batch["exp"][0]:
                trainer.model.reset_running_bbox()
                trainer.current_sentence = batch["exp"][0]
            batch = to_device(batch)
            encoder_pred = trainer.model.resnet(batch, identity_features=None)
            if encoder_pred is None:
                continue
            B = len(batch['flame_path'])
            for ib in range(B):
                flame_path = batch['flame_path'][ib]
                assert '/flame_param/' in flame_path
                out_path = flame_path.replace('/flame_param/', '/deca_param/')
                deca_pred_np = dict([(k, v[ib].cpu().numpy()) for k, v in encoder_pred.deca.items()])
                Path(out_path).parent.mkdir(exist_ok=True, parents=True)
                np.savez(out_path, **deca_pred_np)




if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()
    train(config)
