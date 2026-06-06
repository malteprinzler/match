

import os
from pathlib import Path
import sys
import torch

from loguru import logger
from omegaconf import OmegaConf


torch.backends.cudnn.benchmark = True
import pudb
import copy
import numpy as np
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append('third_party/GEM')
from data.base import DatasetMode
from lib.regressor.model import RegressorModel
from utils.general import build_dataset, build_loader, seed_everything, to_device
from lib.common import slice_dict, dict_to_numpy

def save_face_features(face_features_, out_path):
    """Helper function to save face features to disk."""
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    np.savez(out_path, **face_features_)

def predict_face_features(config):

    train_dataset = build_dataset(config)
    train_loader = build_loader(train_dataset, **{**config.train, **config.data})

    val_config = copy.deepcopy(config)
    val_config.data.join_configs = False
    val_dataset = build_dataset(val_config, camera_list=[val_config.data.test_camera], mode=DatasetMode.validation)
    val_loader = build_loader(val_dataset, **{**config.train, **config.data})

    test_config = copy.deepcopy(config)
    test_config.data.join_configs = False
    test_dataset = build_dataset(test_config, camera_list=[test_config.data.test_camera], mode=DatasetMode.test)
    test_loader = build_loader(test_dataset, **{**config.train, **config.data})

    if config.data_cross_config is None:
        cross_loader = None
    else:
        cross_config = copy.deepcopy(config)
        cross_config.data.join_configs = True
        source_config = OmegaConf.load(cross_config.data_cross_config)
        cross_dataset = build_dataset(cross_config, camera_list=[cross_config.data.test_camera], source_config=source_config, mode=DatasetMode.validation)
        cross_loader = build_loader(cross_dataset, **{**config.train, **config.data})

    regressor = RegressorModel(config, train_dataset)
    regressor.bg_color = "white"

    name_loader_istrain_suffix = [
        ('train', train_loader, True, '_train'), 
        ('val', val_loader, False, '_val'), 
        ('test', test_loader, False, '_val'), 
        ]
    if cross_loader is not None:
        name_loader_istrain_suffix.append(('cross', cross_loader, False, '_val'))

    # Create thread pool executor for asynchronous saving
    max_workers = getattr(config, 'save_workers', 4)  # Default to 4 workers if not specified
    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = []

    current_sentence = None
    for loader_name, loader, train, suffix in name_loader_istrain_suffix:
        if train:
            regressor.train()
        else:
            regressor.eval()

        for i, batch in enumerate(tqdm.tqdm(loader, desc=f'Predicting face features for {loader_name} Dataset')):
            batch = to_device(batch)
            if not train: 
                assert len(batch['exp']) == 1
                if current_sentence != batch["exp"][0]:
                    regressor.reset_running_bbox()
                    current_sentence = batch["exp"][0]
            encoder_pred = regressor.resnet(batch, identity_features=None)
            assert len(batch['exp']) == 1, 'BS must be 1 otherwise working samples may get skipped'            
            if encoder_pred is None:
                continue
            face_features = encoder_pred.pretrained_face_features
            for ib in range(len(batch['image_path'])):                
                image_path = batch['image_path'][ib]
                assert '/images/' in image_path                
                out_path = image_path.replace('/images/', '/face_features/').replace('.jpg', f'{suffix}.npz').replace('.png', f'{suffix}.npz')
                face_features_ = dict_to_numpy(slice_dict(face_features, ib))
                # Submit save task to worker pool
                future = executor.submit(save_face_features, face_features_, out_path)
                futures.append(future)
    
    # Wait for all save operations to complete
    logger.info(f"Waiting for {len(futures)} save operations to complete...")
    for future in tqdm.tqdm(as_completed(futures), total=len(futures), desc='Saving face features'):
        future.result()  # This will raise any exceptions that occurred
    
    executor.shutdown(wait=True)
    logger.info("All face features saved successfully.")


if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()
    predict_face_features(config)
