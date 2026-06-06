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

import argparse
from option_handler.base_train_options import BaseTrainOptions, json_dict

PRE_TRAINING_VERTEX_GROUP_WEIGHTS = {
'fitting_region_skin': 3.0, 'ears': 3.0, 'eyes': 3.0, 'mouth': 3.0
}

S2M_POINT_MASK_WEIGHTS = {
    'w_point_face': 0.0,
    'w_point_ears': 0.0,
    'w_point_eyeballs': 1.0,
    'w_point_eye_region': 0.0,
    'w_point_lips': 0.0,
    'w_point_neck': 0.0,
    'w_point_nostrils': 0.0,
    'w_point_scalp': 0.0,
    'w_point_boundary': 0.0,
}

EDGE_MASK_WEIGHTS = {
    'w_edge_face': 10.0,
    'w_edge_ears': 3.0,
    'w_edge_eyeballs': 50.0,
    'w_edge_eye_region': 10.0,
    'w_edge_lips': 10.0,
    'w_edge_neck': 1.0,
    'w_edge_nostrils': 10.0,
    'w_edge_scalp': 1.0,
    'w_edge_boundary': 10.0,
}

class TrainOptions(BaseTrainOptions):
    def __init__(self):
        super().__init__()

    def initialize(self):
        BaseTrainOptions.initialize(self)
        self.isTrain = True
        return self.parser

    def initialize_extra(self):
        self.add_arg( cate='base', abbr='cf',  name='config-filename', type=str, default='')
        self.add_arg( cate='base', abbr='md',  name='model-directory', type=str, default='./runs/coarse')
        self.add_arg( cate='base', abbr='eid', name='experiment-id', type=str, default='TEMPEH')
        self.add_arg( cate='base', abbr='outpath', name='outpath', type=str, default=argparse.SUPPRESS)

        # data
        self.add_arg( cate='data', abbr='num-spnts', name='number-sample-points', type=list, default=[5741])
        self.add_arg( cate='data', abbr='templ-name', name='template-fname', type=str, default='assets/ava256/face_topology_cleaned.obj')
        self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='stereo_images') 
        # self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='color_images') 
        self.add_arg( cate='data', abbr='b-sigma', name='brightness-sigma', type=float, default=0.33) 

        self.add_arg( cate='data', abbr='data-dir', name='dataset-directory', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='subjects', name='subjects', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='camera-angles', name='camera-angles', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='height', name='height', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='width', name='width', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='frame-stride', name='frame-stride', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='max-captures', name='max-captures', type=int, default=argparse.SUPPRESS)

        # nersemble dataset
        self.add_arg( cate='data', abbr='nersemble-dir', name='nersemble-dataset-directory', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-subjects', name='nersemble-subjects', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-camera-angles', name='nersemble-camera-angles', type=str, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-frame-stride', name='nersemble-frame-stride', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-height', name='nersemble-height', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-width', name='nersemble-width', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='data', abbr='nersemble-max-captures', name='nersemble-max-captures', type=int, default=argparse.SUPPRESS)

        # uv rendering
        self.add_arg( cate='uv', abbr='uv-h',  name='uv-out-height', type=int, default=argparse.SUPPRESS)    
        self.add_arg( cate='uv', abbr='uv-w',  name='uv-out-width', type=int, default=argparse.SUPPRESS)   
        self.add_arg( cate='uv', abbr='uv-nsamples',  name='uv-n-samples', type=int, default=argparse.SUPPRESS)   


        # train
        self.add_arg( cate='train', abbr='niter',  name='num-iterations', type=int, default=argparse.SUPPRESS)  # global
        self.add_arg( cate='train', abbr='print-freq',  name='print-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='vis-freq',  name='visualize-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='val-freq',  name='validate-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='val-steps',  name='validation-steps', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='save-freq',  name='save-frequency', type=int, default=argparse.SUPPRESS)

        # model
        self.add_arg( cate='model', abbr='gfeature-fusion', name='global-feature-fusion', type=str, default='filtered_mean_var') 
        # self.add_arg( cate='model', abbr='gtrafo', name='global-spatial-transformer', type=str, default='none') # baseline
        self.add_arg( cate='model', abbr='gtrafo', name='global-spatial-transformer', type=str, default='rigid_transformer') # TEMPEH


        self.add_arg( cate='model', abbr='gtrafo-dim', name='global-spatial-transformer-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='gtrafo-levels', name='global-transformer-levels', type=int, default=2)
        self.add_arg( cate='model', abbr='gtrafo-sfactor', name='global-transformer-scale-factor', type=float, default=0.8)
        self.add_arg( cate='model', abbr='gtrafo-pooling', name='global-transformer-use-pooling', type=bool, default=True)

        self.add_arg( cate='model', abbr='sviews', name='sample-views', type=bool, default=True)   
        self.add_arg( cate='model', abbr='min-views', name='minimum-sample-views', type=int, default=12)
        self.add_arg( cate='model', abbr='irf', name='image-resize-factor', type=int, default=8) 
        self.add_arg( cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=8) 
        self.add_arg( cate='model', abbr='feat-arch', name='feature-arch', type=str, default='custom_unet')

        # self.add_arg( cate='model', abbr='global-arch', name='global-arch', type=str, default='v2v') # baseline
        self.add_arg( cate='model', abbr='global-arch', name='global-arch', type=str, default='v2v2') # TEMPEH
        self.add_arg( cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')

        # volumetric sparse point net
        self.add_arg( cate='model', abbr='gvd', name='global-voxel-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='gvi', name='global-voxel-inc', type=float, default=15.0)
        self.add_arg( cate='model', abbr='go', name='global-origin', type=list, default=[0.0, 0.0, 0.0])
        self.add_arg( cate='model', abbr='nm', name='norm', type=str, default="in")

        # loss functions
        self.add_arg( cate='model', abbr='vertex-group-weights', name='vertex-group-weights', type=dict, default=argparse.SUPPRESS)
        self.add_arg( cate='model', abbr='vert-group-path', name='vertex-group-path', type=str, default='assets/ava256/vertex_groups.json')

        # loss weights
        self.add_arg( cate='model', abbr='wp2p', name='weight-p2p', type=float, default=argparse.SUPPRESS) 
        self.add_arg( cate='model', abbr='wnormal', name='weight-normal', type=float, default=argparse.SUPPRESS) # pre-training: 0.0, afterwards: 10.0

        self.initialize_default_parameters()

    def initialize_default_parameters(self):
        self.default_parameters = {
            'num-iterations': 1_000_000,
            'weight-p2p': 1.0,
            'weight-normal': 1.0,
            'vertex-group-weights': PRE_TRAINING_VERTEX_GROUP_WEIGHTS,
            'print-frequency': 100,
            'visualize-frequency': 10_000,
            'validate-frequency': 10_000,
            'validation-steps': 100,
            'save-frequency': 10_000,
            'outpath': None,
            'uv-out-height': None,
            'uv-out-width': None,
            'dataset-directory': None,
            'camera-angles': None,
            'height': None,
            'width': None,
            'frame-stride': None,
            'max-captures': None,
            'uv-n-samples': -1,
            'subjects': None,
            'nersemble-dataset-directory': None,
            'nersemble-subjects': None,
            'nersemble-camera-angles': None,
            'nersemble-frame-stride': None,
            'nersemble-height': None,
            'nersemble-width': None,
            'nersemble-max-captures': None,
        }