# python match/data/ava256_dataset_cli.py
import pudb
from match.data.ava256_dataset import AvaMultiCaptureDataset
from match.utils.data_util import visualize_sample, TempehBatch, MatchBatch
import tqdm
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import sys

if __name__ == '__main__':

    ###
    # Get unique cameras 
    ###
     ###
    dataset = AvaMultiCaptureDataset(
        root_path='/fast/mprinzler/gintern/datasets/ava-256',
        height=786,
        width=512,
        max_captures=10,
        camera_angles = [[0,0], [0, -15],
                  [20,0], [20, 36], [20, -15],
                  [-20,0], [-20, 36], [-20, -15],
                  [40,-5], [40, 25],
                  [-40,-5], [-40, 25],
                ],
        training=None,
        deterministic_shuffle=True,
        process_idx = 0,
        world_size=1,
        # sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
        # coarse_mesh_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
        # uv_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
        frame_stride = 1,
        exclude_subjects =list(),
        only_subjects = list(['APP152']),
        invalid_captures_path = None, 
        head_crop = False,
        require_verts = False,
        require_segmentation=False,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)
    all_cameras = set()
    for i,batch in enumerate(tqdm.tqdm(dataloader)):
        if i > 100:
            break
        all_cameras = all_cameras.union(set(batch['camera'].flatten().tolist()))
    print(f'Used cameras: {len(all_cameras)}', sorted(list(all_cameras)))
    exit()
    ###
    # compare tempeh and ava256->tempeh datasets
    ###
    dataset = AvaMultiCaptureDataset(
        root_path='/is/cluster/mprinzler/projects/gintern/match/data/ava-256_12cams',
        height=786,
        width=512,
        max_captures=10,
        camera_angles = [[0,0], [0, -15],
                  [20,0], [20, 36], [20, -15],
                  [-20,0], [-20, 36], [-20, -15],
                  [40,-5], [40, 25],
                  [-40,-5], [-40, 25],
                ],
        training=False,
        deterministic_shuffle=False,
        process_idx = 0,
        world_size=1,
        # sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
        # coarse_mesh_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
        # uv_directory='/fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/coarse__tempeh_ava256_scale08_wedgeface10__September01__14-23-23_uvmaps_full',
        frame_stride = 10,
        exclude_subjects =list(),
        # only_subjects = list(['TCE049']),
        invalid_captures_path = None, 
        head_crop = False,
        require_verts = False,
        require_segmentation=False,
    )

    old_sample = dataset[0]
    for i in tqdm.tqdm(range(5,100)):
        print(i)
        pudb.set_trace()
        sample = dataset[i]
        batch = default_collate([sample, old_sample])
        old_sample = sample
    pudb.set_trace()

    # Use DataLoader for multiprocessed data loading
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=16)

    for i, batch in enumerate(tqdm.tqdm(dataloader)):
        pass
        

    tempeh_dataset = TEMPEH_AvaMultiCaptureDataset(
        root_path='/fast/mprinzler/gintern/datasets/ava-256',
        height=786,
        width=512,
        max_captures=10,
        camera_angles_specified = [[0,0], [0, -15],
                  [20,0], [20, 36], [20, -15],
                  [-20,0], [-20, 36], [-20, -15],
                  [40,-5], [40, 25],
                  [-40,-5], [-40, 25],
                ],
        training=False,
        deterministic_shuffle=False,
        process_idx = 0,
        world_size=1,
        sapiens_segmentation_directory='/fast/mprinzler/gintern/datasets/ava-256_sapiens_segmentations/framestride_10/sapiens_1b',
        frame_stride = 10,
        exclude_subjects =list(),
        only_subjects = list(['TCE049']),
        invalid_captures_path = None, 
        head_crop = False,
    )
    for i in range(10):
        tempeh_sample = tempeh_dataset[i]
        match_sample = dataset[i]
        match_tempeh_sample = TempehBatch.from_match_batch(MatchBatch(match_sample).unsqueeze()).squeeze()
        print('sample', i)
        from torchvision.utils import make_grid, save_image
        save_image(make_grid(match_tempeh_sample['stereo_images_augmented']), f'demos/match_tempeh_sample_{i:02d}.png')
        save_image(make_grid(tempeh_sample['stereo_images_augmented']), f'demos/tempeh_sample_{i:02d}.png')
        for k, v in match_tempeh_sample.items():
            if isinstance(v, torch.Tensor):
                rel_dist = (torch.tensor(tempeh_sample[k]) - torch.tensor(match_tempeh_sample[k])).abs()/torch.tensor(tempeh_sample[k]).abs()
                rel_dist = torch.nan_to_num(rel_dist, nan=0.0)
                print(k, 'max relative error:', rel_dist.max())
        print('--------------------------------')
        
    # sample = dataset[0]
    # pudb.set_trace()
    # visualize_sample(sample, 'demos/sample_vis.jpg')