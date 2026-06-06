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


def train(config):
    train_dataset = build_dataset(config)
    train_loader = build_loader(train_dataset, in_order=False, prefetch_factor=10, use_infinite_sampler=True, **{**config.train, **config.data})

    val_config = copy.deepcopy(config)
    val_config.data.join_configs = False
    val_dataset = build_dataset(val_config, camera_list=[val_config.data.test_camera], mode=DatasetMode.validation)

    test_config = copy.deepcopy(config)
    test_config.data.join_configs = False
    test_dataset = build_dataset(test_config, camera_list=[test_config.data.test_camera], mode=DatasetMode.test)

    cross_config = copy.deepcopy(config)
    cross_config.data.join_configs = True
    source_config = OmegaConf.load(cross_config.data_cross_config)
    cross_dataset = build_dataset(cross_config, camera_list=[cross_config.data.test_camera], source_config=source_config, mode=DatasetMode.validation)

    logger.info(f"Training with total of {len(train_dataset)} frames")

    trainer_type = config.train.get("trainer", None)

    if trainer_type is None:
        trainer = ApperanceTrainer(config, train_dataset)
    elif trainer_type.upper() == "REGRESSOR":
        trainer = RegressorTrainer(config, train_dataset)
    else:
        raise RuntimeError("Selected trainer mode is not supported!")
    iterations = config.train.iterations
    trainer.restore(force=True)

    train_iter = iter(train_loader)
    trainer.open()
    trainer.model.bg_color = "random"


    while trainer.curr_iter < iterations + 1:
        neutral_frame_mean_init_iteration = config.train.get('neutral_frame_mean_init_iteration', 0)
        if trainer.curr_iter == 1 and neutral_frame_mean_init_iteration>0:
            mean_frame = trainer.get_mean_frame()
            mean_samples = train_dataset.get_all_frame_samples(mean_frame)
            mean_batch = default_collate(mean_samples)
            trainer.neutral_frame_init_mean(mean_batch)

        try:
            batch = next(train_iter)
        except Exception as e:
            logger.info(f"Iterator {str(e)}")
            train_iter = iter(train_loader)

        trainer.step(batch)


        if isinstance(trainer, RegressorTrainer) and (trainer.curr_iter % config.train.get('eval_n_steps', 2500) == 0 or trainer.curr_iter == iterations+1):
            name_ds_mode_seed = [(f'val', val_dataset, RegressorMode.VAL, 18)]
            if trainer.is_regressor_training:
                name_ds_mode_seed.extend([('test', test_dataset, RegressorMode.TEST, 20), ('cross', cross_dataset, RegressorMode.CROSS, 0)])
            for ds_name, ds, mode, seed in name_ds_mode_seed:
                trainer.set_mode(mode)
                identity_features = None
                if mode == RegressorMode.CROSS:
                    identity_features = trainer.get_canonical_features(ds.source.identity_frame)
                
                # evaluation
                loader = build_loader(
                    ds,
                    batch_size=1,
                    num_workers=8,
                    shuffle=True,
                    persistent_workers=False,
                    seed=seed,
                )
                trainer.run_evaluation(loader, name=f'eval_{ds_name}', identity_features=identity_features, save_images=False)

                if config.train.get('make_videos', True):
                    # make video
                    loader = build_loader(
                        ds,
                        batch_size=1,
                        num_workers=8,
                        shuffle=False,
                        persistent_workers=False,
                    )
                    framestride = 2 if mode == RegressorMode.VAL else 1
                    trainer.make_video(loader, name=f'eval_{ds_name}', identity_features=identity_features, framestride=framestride)



            trainer.train()


    trainer.close()


def folders(config):
    t = config.train
    canon = os.path.join(t.run_dir, "canonical")
    for folder in [t.run_dir, t.ckpt_dir, t.tb_dir, t.progress_dir, canon, t.results_dir]:
        logger.info(f"Creating folder {folder}")
        Path(folder).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    folders(config)
    seed_everything()

    save_cfg_path = f'{config.train.run_dir}/config.yml'
    if os.path.exists(save_cfg_path):
        os.system(f'rm {save_cfg_path}')
    os.system(f'cp {path} {save_cfg_path}')

    train(config)
