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
from option_handler.base_train_options import BaseTrainOptions

# ToFu/ToFu+
# POINT_MASK_WEIGHTS = {
#     'w_point_face': 1.0,
#     'w_point_ears': 1.0,
#     'w_point_eyeballs': 1.0,
#     'w_point_eye_region': 1.0,
#     'w_point_lips': 1.0,
#     'w_point_neck': 1.0,
#     'w_point_nostrils': 1.0,
#     'w_point_scalp': 1.0,
#     'w_point_boundary': 1.0,
# }

# TEMPEH
POINT_MASK_WEIGHTS = {
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
    'w_edge_lips': 10.0,    # 3.0 submitted
    'w_edge_neck': 1.0,
    'w_edge_nostrils': 10.0,
    'w_edge_scalp': 1.0,
    'w_edge_boundary': 10.0 # 3.0 submitted
}

class TrainOptions(BaseTrainOptions):
    def initialize(self):
        BaseTrainOptions.initialize(self)
        self.isTrain = True
        return self.parser

    def initialize_extra(self):
        self.add_arg( cate='base', abbr='cf',  name='config-filename', type=str, default='')
        self.add_arg( cate='base', abbr='md',  name='model-directory', type=str, default='./runs/refinement')
        self.add_arg( cate='base', abbr='eid', name='experiment-id', type=str, default='TEMPEH')
        self.add_arg( cate='base', abbr='outpath', name='outpath', type=str, default=argparse.SUPPRESS)

        # data
        self.add_arg( cate='data', abbr='num-spnts', name='number-sample-points', type=list, default=[5741])
        self.add_arg( cate='data', abbr='templ-name', name='template-fname', type=str, default='/home/mprinzler/projects/gintern/gtempeh/assets/ava256/face_topology_cleaned.obj')
        self.add_arg( cate='data', abbr='vmask-name', name='vertex-mask-fname', type=str, default='./data/template/vertex_masks2_ava.npz')  # TODO fix actual weights
        self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='stereo_images') 
        # self.add_arg( cate='data', abbr='input-img', name='input-image-type', type=str, default='color_images') 
        self.add_arg( cate='data', abbr='b-sigma', name='brightness-sigma', type=float, default=0.33) 
        self.add_arg( cate='data', abbr='v_scan', name='scan-vertex-count', type=str, default=15000)

        # train
        self.add_arg( cate='train', abbr='ni',  name='num-iterations', type=int, default=argparse.SUPPRESS) 
        self.add_arg( cate='train', abbr='print-freq',  name='print-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='vis-freq',  name='visualize-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='val-freq',  name='validate-frequency', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='val-steps',  name='validation-steps', type=int, default=argparse.SUPPRESS)
        self.add_arg( cate='train', abbr='save-freq',  name='save-frequency', type=int, default=argparse.SUPPRESS)    

        # uv rendering
        self.add_arg( cate='uv', abbr='uv-h',  name='uv-out-height', type=int, default=argparse.SUPPRESS)    
        self.add_arg( cate='uv', abbr='uv-w',  name='uv-out-width', type=int, default=argparse.SUPPRESS)    
            

        # model
        # self.add_arg( cate='model', abbr='lfeature-fusion', name='local-feature-fusion', type=str, default='mean_var') # baseline
        self.add_arg( cate='model', abbr='lfeature-fusion', name='local-feature-fusion', type=str, default='visibility_filtered_normal_weighted_mean_var_v2') # TEMPEH

        self.add_arg( cate='model', abbr='ltrafo', name='local-spatial-transformer', type=str, default='none')  

        self.add_arg( cate='model', abbr='ltrafo-dim', name='local-spatial-transformer-dim', type=int, default=32)
        self.add_arg( cate='model', abbr='ltrafo-levels', name='local-transformer-levels', type=int, default=2)
        self.add_arg( cate='model', abbr='ltrafo-sfactor', name='local-transformer-scale-factor', type=float, default=1.0)

        self.add_arg( cate='model', abbr='sviews', name='sample_views', type=bool, default=True)
        self.add_arg( cate='model', abbr='min-views', name='minimum-sample-views', type=int, default=8)        
        self.add_arg( cate='model', abbr='irf', name='image-resize-factor', type=int, default=4)
        self.add_arg( cate='model', abbr='desc-dim', name='descriptor-dim', type=int, default=8) 
        self.add_arg( cate='model', abbr='feat-arch', name='feature-arch', type=str, default='uresnet2')    # TEMPEH/ToFu/ToFu+

        self.add_arg( cate='model', abbr='local-arch', name='local-arch', type=str, default='v2v')
        self.add_arg( cate='model', abbr='pretr-path', name='pretrained-path', type=str, default='')

        # volumetric sparse point net
        self.add_arg( cate='model', abbr='get', name='global-embedding-type', type=str, default='coords')
        self.add_arg( cate='model', abbr='lvd', name='local-voxel-dim', type=int, default=8)
        self.add_arg( cate='model', abbr='lvi', name='local-voxel-inc-list', type=list, default=[2.0])
        self.add_arg( cate='model', abbr='nm', name='norm', type=str, default="bn")

        # loss weights
        self.add_arg( cate='model', abbr='wpoints', name='weight-points-recon', type=float, default=1.0)
        self.add_arg( cate='model', abbr='wpointm', name='point-mask-weights', type=dict, default=POINT_MASK_WEIGHTS)
        self.add_arg( cate='model', abbr='wp2s', name='weight-points2surface', type=float, default=10.0)  # submission: 30.0, TEMPEH: 10, ToFu/ToFu+: 0.0
        self.add_arg( cate='model', abbr='wtex', name='weight-texture-recon', type=float, default=0.0)   # TEMPEH: 0.0

        self.add_arg( cate='model', abbr='gmo', name='gmo_sigma', type=float, default=10.0)  # TEMPEH: 10.0
        self.add_arg( cate='model', abbr='wedge', name='weight-edge-regularizer', type=float, default=0.3) # TEMPEH: 0.3, ToFu/ToFu+: 0.0
        self.add_arg( cate='model', abbr='wedgem', name='edge-mask-weights', type=dict, default=EDGE_MASK_WEIGHTS)

        self.initialize_default_parameters()

    def initialize_default_parameters(self):
        self.default_parameters = {
            'num-iterations': 150000,
            'global-model-root-dir': '',
            'print-frequency': 100,
            'visualize-frequency': 5000,
            'validate-frequency': 5000,
            'validation-steps': 100,
            'save-frequency': 5000,
            'global-registration-root-dir': '',
            'outpath': None,
            'uv-out-height': None,
            'uv-out-width': None,
        }

