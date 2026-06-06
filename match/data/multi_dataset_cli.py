# python src/data/multi_dataset_cli.py
from match.data import MultiDataLoader
from torch.utils.data.dataloader import DataLoader
from match.data.data_utils import visualize_sample_wojciech, visualize_sample, visualize_camera_grid
from src_tempeh.utils.visualize_predictions import visualize_predictions
from src_tempeh.utils.data_utils import gtempeh_batch_to_tempeh_batch
from match.utils import file_util
from src_tempeh.utils import mesh_helper
from PIL import Image
import pudb
import numpy as np
import torch
import tqdm
import einops
if __name__ == '__main__':
    # root = '/is/rg/ncs/datasets/ava-256'
    # cam_ids = ['401045', '400943', '400944', '400948', '400951', '400952',
    #            '400938', '400949', '401627', '401630', '401632', '401633',  # alternative setup
    #            ]
    # cam_angles = None
    # cam_ids = None
    max_captures = 1
    dataloader = MultiDataLoader(dataloader_kwargs=dict(shuffle=True, batch_size=3),
                                 dataset_configs=[
                                    ('ava256', dict(
                                        root_path='/fast/mprinzler/gintern/datasets/ava-256', 
                                        max_captures=max_captures, 
                                        # cameras_specified=cam_ids, 
                                        # frames_per_subject=1, 
                                        # deterministic_shuffle=True,
                                        # stage1_directory='/fast/mprinzler/gintern/datasets/ava-256_gtuvmaps_framestride10/ava_framestride10_correctuv',
                                        sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
                                        uv_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/easyava256_tempeh_860k_framestride10/ava',
                                        stage1_directory = '/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/easyava256_tempeh_860k_framestride10/ava',
                                        use_gt_uvmaps=False,
                                        camera_angles_specified=[[0,0], [0, 36], [0, -15],
                                            [20,0], [20, 36], [20, -15],
                                            [-20,0], [-20, 36], [-20, -15],
                                            [40,-5], [40, 25],
                                            [-40,-5], [-40, 25],
                                          ], 
                                        deterministic_shuffle=True,
                                        training=True,
                                        # max_captures=1,
                                        frame_stride = 10,
                                        # exclude_subjects = ['FXN596'],
                                        height=786, width=512,
                                        head_crop=True,
                                        head_crop_height=512,
                                        head_crop_width=512,
                                    )),
                                    ('nersemble', dict(
                                        root_path = '/fast/mprinzler/gintern/datasets/nersemble_tracked_extracted',
                                        # frames_per_subject=1, 
                                        # deterministic_shuffle=True,
                                        stage1_directory='/fast/mprinzler/gintern/datasets/nersemble_uvpredictions/nersemble_tempehonava_1M_framestride4/nersemble',
                                        uv_directory = '/fast/mprinzler/gintern/datasets/nersemble_uvpredictions/nersemble_tempehonava_1M_framestride4/nersemble',
                                        camera_angles_specified=[  
                                            [-49,15], [-28, 15], [-6, 15], [6, 15], [28, 15], [49, 15],
                                            [-49,-15], [-28, -15], [-6, -15], [6, -15], [28, -15], [49, -15],
                                            [-16,13]], 
                                        deterministic_shuffle=True,
                                        training=True,
                                        frame_stride = 4,
                                        width=550, height=802,
                                        head_crop=True,
                                        head_crop_height=512,
                                        head_crop_width=512,
                                        max_captures = max_captures*8,
                                        invalid_captures_path = '/home/mprinzler/projects/gintern/gtempeh/assets/nersemble/invalid_captures.txt',
                                        verts_pad_to_n = 5741
                                    ))
                                 ])
    for i, batch in enumerate(dataloader):
       Image.fromarray(np.round(einops.rearrange(batch['image'], 'b v c h w -> (b h) (v w) c').cpu().numpy()*255).astype(np.uint8)).save(f'demos/mixed_{i:02d}.jpg')
       print(i)
       pudb.set_trace()
       print()