# python match/data/nersemble_dataset_cli.py
import pudb
from match.data.nersemble_dataset import NersembleMultiCaptureDataset
from match.data.ava256_dataset import AvaMultiCaptureDataset
from match.utils.data_util import visualize_sample, TempehBatch, MatchBatch, visualize_camera_grid

if __name__ == '__main__':

    ###
    # visualize nersemble sample
    ###

    dataset = NersembleMultiCaptureDataset(
        root_path='/fast/mprinzler/gintern/datasets/nersemble_tracked_extracted',
        width=550, height=802,
        head_crop=True,
        # head_crop_height=512,
        head_crop_height=512+128,
        head_crop_width=512,
        head_crop_offset_y = -64,
        invalid_captures_path = 'assets/nersemble/invalid_captures.txt',
        coarse_mesh_directory='/fast/mprinzler/gintern/datasets/mixed_uvpredictions/MATCH_TEMPEH_fixednersemblealign/nersemble',
        uv_directory='/fast/mprinzler/gintern/datasets/mixed_uvpredictions/MATCH_TEMPEH_fixednersemblealign/nersemble',
        max_captures=1,
        camera_angles = [  [-49,15], [-28, 15], [-16, 13], [5, 14], [27, 15], [48, 16],
  [-48,-15], [-16, -16], [-6, -16], [5, -17], [26, -15], [47, -16],
                ],
        training=None,
        deterministic_shuffle=False,
        process_idx = 0,
        world_size=1,
        frame_stride = 4,
    )
    sample = dataset[0]
    visualize_sample(sample, 'demos/sample_vis_nersemble.jpg')

    # ###
    # # compare ava256 and nersemble datasets
    # ###
    # dataset = NersembleMultiCaptureDataset(
    #     root_path='/fast/mprinzler/gintern/datasets/nersemble_tracked_extracted',
    #     width=550, height=802,
    #     head_crop=True,
    #     # head_crop_height=512,
    #     head_crop_height=512+128,
    #     head_crop_width=512,
    #     head_crop_offset_y = -64,
    #     invalid_captures_path = 'assets/nersemble/invalid_captures.txt',
    #     max_captures=10,
    #     camera_angles = [  
    #                 [-49,15], [-28, 15], [-16, 13], [5, 14], [27, 15], [48, 16],
    #                 [-48,-15], [-16, -16], [-6, -16], [5, -17], [26, -15], [47, -16],
    #             ],
    #     training=None,
    #     deterministic_shuffle=True,
    #     process_idx = 0,
    #     world_size=1,
    #     frame_stride = 4,
    # )

    # dataset_ava256 = AvaMultiCaptureDataset(
    #     root_path='/fast/mprinzler/gintern/datasets/ava-256',
    #     height=786,
    #     width=512,
    #     max_captures=10,
    #     camera_angles = [[0,0], [0, -15],
    #               [20,0], [20, 36], [20, -15],
    #               [-20,0], [-20, 36], [-20, -15],
    #               [40,-5], [40, 25],
    #               [-40,-5], [-40, 25],
    #             ],
    #     training=None,
    #     deterministic_shuffle=True,
    #     process_idx = 0,
    #     world_size=1,
    #     # sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
    #     # coarse_mesh_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
    #     # uv_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
    #     frame_stride = 1,
    #     exclude_subjects =list(),
    #     only_subjects = list(['APP152']),
    #     invalid_captures_path = None, 
    #     head_crop = False,
    #     require_verts = False,
    #     require_segmentation=False,
    # )

    # visualize_camera_grid(dict(ava256=dataset_ava256[0], nersemble=dataset[0]), 'demos/camera_grid_comparison.html')

