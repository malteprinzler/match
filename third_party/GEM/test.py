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
import pudb
from lib.apperance.trainer import ApperanceTrainer
from lib.regressor.trainer import RegressorTrainer, RegressorMode
from utils.general import build_dataset, build_loader, seed_everything, to_device
torch.backends.cudnn.benchmark = True

from torch.utils.data._utils.collate import default_collate
import copy
from data.base import DatasetMode


def test(config):
    train_dataset = build_dataset(config)
    
    test_config = copy.deepcopy(config)
    test_config.data.join_configs = False
    test_dataset = build_dataset(test_config, camera_list=[test_config.data.test_camera], mode=DatasetMode.test)

    cross_config = copy.deepcopy(config)
    cross_config.data.join_configs = True
    source_config = OmegaConf.load(cross_config.data_cross_config)
    cross_dataset = build_dataset(cross_config, camera_list=[cross_config.data.test_camera], source_config=source_config, mode=DatasetMode.test)

    trainer = RegressorTrainer(config, train_dataset)
    trainer.restore(force=True)
    trainer.eval()


    name_ds_mode_seed = [
        ('test', test_dataset, RegressorMode.TEST, 0), 
        ('cross', cross_dataset, RegressorMode.CROSS, 0)
        ]
    for ds_name, ds, mode, seed in name_ds_mode_seed:
        trainer.set_mode(mode)
        identity_features = None
        if mode == RegressorMode.CROSS:
            identity_features = trainer.get_canonical_features(str(ds.source.parse(ds.source.identity_frame['file_path'])))
        
        # evaluation
        loader = build_loader(
            ds,
            batch_size=1,
            num_workers=8,
            shuffle=False,
            persistent_workers=False,
            seed=seed,
        )
        trainer.save_val_predictions(loader, name=f'{ds_name}', identity_features=identity_features, mode=mode)





if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()
    test(config)
