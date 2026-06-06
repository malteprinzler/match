"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
Using this computer program means that you agree to the terms 
in the LICENSE file included with this software distribution. 
Any use not explicitly granted by the LICENSE is prohibited.

Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

For comments or questions, please email us at tempeh@tue.mpg.de
"""

import os
from os.path import join
import time
import math
import numpy as np
import argparse
import imageio
from glob import glob
import torch
import torch.nn.parallel
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict
from psbody.mesh import Mesh
from .scheduler import LinearConstantDecayScheduler
from utils.utils import get_filename

# -----------------------------------------------------------------------------

class BaseTrainer():

    def __init__(self, args):
        self.args = args
        self.global_step = 0
        self.training = True

    def initialize(self, init_full_dataset=False, process_idx=0, world_size=1):
        self.control_seeds()
        self.mkdirs()
        self.register_template_mesh()
        self.register_mesh_sampler()
        self.register_model()
        self.register_losses()
        self.register_optimizer()
        self.register_scheduler()
        self.resume_checkpoint()
        self.register_dataset(init_full_dataset=init_full_dataset, process_idx=process_idx, world_size=world_size)
        self.register_logger()
        self.register_visualizer()

    # ----------------------
    # meta

    def control_seeds(self):
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        np.random.seed(self.args.seed)

    def mkdirs(self):
        if self.args.model_directory == '': raise RuntimeError(f'invalid model_directory = {self.args.model_directory}')
        if self.args.experiment_id == '': raise RuntimeError(f'invalid experiment_id = {self.args.experiment_id}')
        self.directory_output = join(self.args.model_directory, self.args.experiment_id)
        os.makedirs(self.directory_output, exist_ok=True)

        self.model_dir = join(self.directory_output, 'checkpoints')
        os.makedirs(self.model_dir, exist_ok=True)

    # ----------------------
    # model

    def register_template_mesh(self):
        template_fname = self.args.template_fname
        template_mesh = Mesh(filename=template_fname)
        template_mesh.v[:] *= 1000
        self.template_verts = template_mesh.v
        self.template_faces = template_mesh.f
        self.template_face_uv_coords = template_mesh.vt[template_mesh.ft]

    def register_mesh_sampler(self):
        raise NotImplementedError(f"mesh sampler not registered")

    def register_model(self):
        raise NotImplementedError(f"model not yet registered")

    def register_losses(self):
        raise NotImplementedError(f"model not yet registered")

    def set_train(self):
        self.model.train(True)
        self.training = True

    def set_eval(self):
        self.model.train(False)
        self.training = False

    def set_test(self):
        self.set_eval()

    def save_checkpoint(self):
        model_path = os.path.join(self.model_dir, 'model_%08d.pth' % (self.global_step))
        torch.save({
            'model': self.model.module.state_dict(),
            'optimizer_model': self.optimizer_model.state_dict(),
            'scheduler_model': self.scheduler_model.state_dict()
            }, model_path)

    def resume_checkpoint(self): 
        model_paths = sorted(glob(join(self.model_dir, '*.pth')))
        print(f"resume_checkpoint(): found {len(model_paths)} models")
        if len(model_paths) > 0:
            # pick the latest one
            resume_path = model_paths[-1]
            start_iteration =  int(get_filename(resume_path)[6:])
            self.global_step = start_iteration+1

            # load
            try:
                state_dicts = torch.load(resume_path)
                self.model.module.load_state_dict(state_dicts['model'])
                self.optimizer_model.load_state_dict(state_dicts['optimizer_model'])
                if 'scheduler_model' in state_dicts:
                    self.scheduler_model.load_state_dict(state_dicts['scheduler_model'])
                else:
                    print('No scheduler checkpoint found, starting from scratch')
                print('Resuming progress from %s iteration' % self.global_step)
                print(f"\tfrom model path {resume_path}")
            except Exception as e:
                self.model.load(resume_path)
                # ignore loading optimizer info
                print('(WORKAROUND) Resuming progress from %s iteration' % self.global_step)
                print(f"\tfrom model path {resume_path}")

    # ----------------------
    # dataset

    def worker_init_fn(self, worker_id):
        # to properly randomize:
        # https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def make_data_loader(self, dataset, cuda=True, shuffle=True, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.args.batch_size
        default_kwargs = {'num_workers': self.args.thread_num, 'pin_memory': True} if cuda else {}
        kwargs = {**default_kwargs, **kwargs}
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True,
                                            worker_init_fn=self.worker_init_fn, **kwargs)

    def register_dataset(self):
        raise NotImplementedError(f"dataset not yet registered")

        # example
        self.dataset_train = None
        self.dataloader_train = None
        self.dataset_val = None
        self.dataloader_val = None

    # ----------------------
    # optimizer

    def register_optimizer(self):
        self.optimizer_model = torch.optim.AdamW([
                                    {'params': self.model.parameters(), 'lr': self.args.learning_rate}, ])

    def register_scheduler(self):
        self.scheduler_model = LinearConstantDecayScheduler(
            optimizer=self.optimizer_model,
            linear_steps=self.args.lr_linear_steps,
            constant_steps=self.args.lr_constant_steps,
            decay_steps=self.args.lr_decay_steps,
            decay_rate=self.args.lr_decay_rate,
            min_multiplier=self.args.lr_min_multiplier
        )    
   

    # ----------------------
    # logger

    def register_logger(self):
        # from utils.simple_logger import Logger
        from utils.tensorboard_logger import Logger as TensorboardLogger

        # tensorboard logger
        self.tb_logger = TensorboardLogger(os.path.join(self.directory_output, 'logs'))

    # ----------------------
    # visualizer

    def register_visualizer(self):
        self.visualizer = None

    # ----------------------
    # main components

    def feed_data(self, data, mode="train"):
        # consumes a data instance from data loader to reorganize for model.forward format
        # produces self.data and self.inputs as dict
        raise NotImplementedError(f"feed_data() not defined yet")

    def forward(self):
        # consumes self.inputs
        # produces self.predicted
        raise NotImplementedError(f"forward() not defined yet")

    def compute_losses(self):
        # consumes self.inputs, self.data, self.predicted
        # produces self.loss (one single scalar loss to be optimized) and other losses during training
        # also records the losses
        raise NotImplementedError(f"compute_losses() not defined yet")

    def backward(self):
        raise NotImplementedError(f"backward() not defined yet")

    def save_visualizations(self, demo_id, mode='train'):
        # produces and saves visualization
        raise NotImplementedError(f"save_visualizations() not defined yet")

    # ----------------------
    # main processes

    def train_one_epoch(self):
        pass

    def validate(self):
        pass

    def run(self):
        pass
