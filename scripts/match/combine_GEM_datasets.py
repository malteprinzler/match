# 
# Toyota Motor Europe NV/SA and its affiliated companies retain all intellectual 
# property and proprietary rights in and to this software and related documentation. 
# Any commercial use, reproduction, disclosure or distribution of this software and 
# related documentation without an express license agreement from Toyota Motor Europe NV/SA 
# is strictly prohibited.
#


from typing import Optional, Literal, List
from copy import deepcopy
import json
from pathlib import Path
import shutil
import random
import argparse
import gin
from match.utils import gin_util


class NeRFDatasetAssembler:
    def __init__(self, src_root: Path, out_folder: Path, val_cam_idx: int, division_mode: Literal['random_single', 'random_group', 'last', 'explicit']='random_group', src_folders_test=None, skip=[]):
        self.src_folders = [p for p in sorted(src_root.iterdir()) if (not any([p.name.startswith(s) for s in skip])) and (not p.name.startswith("UNION"))] 
        print(f'SRC FOLDERS: {[p.name for p in self.src_folders]}')
        self.tgt_folder = out_folder
        self.num_timestep = 0
        self.val_cam_idx = val_cam_idx

        for src_folder in self.src_folders:
            assert src_folder.exists(), f"Error: could not find {src_folder}"
            assert src_folder.parent == out_folder.parent, "All source folders must be in the same parent folder as the target folder"

        # use the subject name as the random seed to sample the test sequence
        subject = src_root.name
        random.seed(subject)

        if division_mode == 'random_single':
            self.src_folders_test = [self.src_folders.pop(int(random.uniform(0, 1) * len(self.src_folders)))]
        elif division_mode == 'random_group':
            # sample one sequence as the test sequence every `group_size` sequences
            self.src_folders_test = []
            num_all = len(self.src_folders)
            group_size = 10
            num_test = max(1, num_all // group_size)
            indices_test  = []
            for gi in range(num_test):
                idx = min(num_all - 1, random.randint(0, group_size - 1) + gi * group_size)
                indices_test.append(idx)

            for idx in indices_test:
                self.src_folders_test.append(self.src_folders.pop(idx))
        elif division_mode == 'last':
            self.src_folders_test = [self.src_folders.pop(-1)]
        elif division_mode == 'explicit':
            assert src_folders_test is not None
            self.src_folders_test = [p for p in self.src_folders if p.name in src_folders_test]
            self.src_folders = [p for p in self.src_folders if p not in self.src_folders_test]

        else:
            raise ValueError(f"Unknown division mode: {division_mode}")

        self.src_folders_train = self.src_folders

    def write(self):
        self.combine_dbs(self.src_folders_train, division='train')
        self.combine_dbs(self.src_folders_test, division='test')

    def combine_dbs(self, src_folders, division: Optional[Literal['train', 'test']] = None):
        db = None
        for i, src_folder in enumerate(src_folders):
            dbi_path = src_folder / "transforms.json"
            assert dbi_path.exists(), f"Could not find {dbi_path}"
            # print(f"Loading database: {dbi_path}")
            dbi = json.load(open(dbi_path, "r"))
           
            dbi['timestep_indices'] = [t + self.num_timestep for t in dbi['timestep_indices']]
            self.num_timestep += len(dbi['timestep_indices'])
            for frame in dbi['frames']:
                # drop keys that are irrelevant for a combined dataset
                frame.pop('timestep_index_original', None)
                frame.pop('timestep_id', None)

                # accumulate timestep indices
                frame['timestep_index'] = dbi['timestep_indices'][frame['timestep_index']]

                # complement the parent folder
                frame['file_path'] = str(Path('..') / Path(src_folder.name) / frame['file_path'])
                if 'flame_param_path' in frame:
                    frame['flame_param_path'] = str(Path('..') / Path(src_folder.name) / frame['flame_param_path'])
                if 'fg_mask_path' in frame:
                    frame['fg_mask_path'] = str(Path('..') / Path(src_folder.name) / frame['fg_mask_path'])
            
            if db is None:
                db = dbi
            else:
                db['frames'] += dbi['frames']
                db['timestep_indices'] += dbi['timestep_indices']
            
        if not self.tgt_folder.exists():
            self.tgt_folder.mkdir(parents=True)
        
        if division == 'train':
            # leave one camera for validation
            db_train = {k: v for k, v in db.items() if k not in ['frames', 'camera_indices']}
            db_train['frames'] = []
            db_val = deepcopy(db_train)

            if len(db['camera_indices']) > 1 and self.val_cam_idx is not None:
                # when having multiple cameras, leave one camera for validation (novel-view sythesis)
                db_train['camera_indices'] = [i for i in db['camera_indices'] if i != self.val_cam_idx]
                db_val['camera_indices'] = [self.val_cam_idx]
            else:
                # when only having one camera, use same camera for train and validation but throw warning
                db_train['camera_indices'] = db['camera_indices']
                db_val['camera_indices'] = [self.val_cam_idx]
                print(f"WARNING: Only one camera found for subject {self.tgt_folder.parent.name}, using same camera for train and validation")

            for frame in db['frames']:
                if frame['camera_index'] in db_train['camera_indices']:
                    db_train['frames'].append(frame)
                if frame['camera_index'] in db_val['camera_indices']:
                    db_val['frames'].append(frame)
                if not (frame['camera_index'] in db_train['camera_indices'] or frame['camera_index'] in db_val['camera_indices']):
                    raise ValueError(f"Unknown camera index: {frame['camera_index']}")
                
            write_json(db_train, self.tgt_folder, 'train')
            write_json(db_val, self.tgt_folder, 'val')

            with open(self.tgt_folder / 'sequences_trainval.txt', 'w') as f:
                for folder in src_folders:
                    f.write(folder.name + '\n')
        else:
            db['timestep_indices'] = sorted(db['timestep_indices'])
            write_json(db, self.tgt_folder, division)

            with open(self.tgt_folder / f'sequences_{division}.txt', 'w') as f:
                for folder in src_folders:
                    f.write(folder.name + '\n')

    
def write_json(db, tgt_folder, division=None):
    fname = "transforms.json" if division is None else f"transforms_{division}.json"
    json_path = tgt_folder / fname
    print(f"Writing database: {json_path}")
    with open(json_path, "w") as f:
        json.dump(db, f, indent=4)
    
@gin.configurable()
def combine_GEM_datasets(
        base_path: Path,
        subjects: List[str],
        val_cam_idx: int,
        division_mode: Literal['random_single', 'random_group', 'last', 'explicit']='random_group',
        skip=[],
        src_folders_test: Optional[List[str]]=None,
        out_folder_name: Optional[str]=None,
    ):
    """Combine GEM datasets for multiple subjects.
    
    Args:
        subjects: List of subject IDs to process
        base_path: Base path containing subject folders
        val_cam_idx: Camera index to use for validation
        division_mode: How to split sequences into train/test
        skip: List of sequence prefixes to skip
        src_folders_test: Explicit list of test sequence names (if division_mode='explicit')
        out_folder_name: Name for output folder. If None, uses 'UNION_GEM_{subject}'
    """
    print("==== Begin assembling datasets ====")
    print(f"Division mode: {division_mode}")
    print(f"Processing {len(subjects)} subjects: {subjects}")

    base_path = Path(base_path)

    for subject in subjects:
        print(f"\n==== Processing subject: {subject} ====")
        src_root = base_path / subject
        
        if out_folder_name is None:
            out_folder = base_path / subject / f"UNION_GEM_{subject}"
        else:
            out_folder = base_path / subject / out_folder_name.format(subject=subject)
        
        nerf_dataset_assembler = NeRFDatasetAssembler(
            src_root=src_root, 
            out_folder=out_folder, 
            val_cam_idx=val_cam_idx, 
            division_mode=division_mode, 
            src_folders_test=src_folders_test, 
            skip=skip
        )
        nerf_dataset_assembler.write()

    print("\nDone!")


def main(argv: list[str]|None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gin_configs", action="append", default=[])
    parser.add_argument("--gin_bindings", action="append", default=[])
    FLAGS = parser.parse_args()

    # explicitly initialize the warm pool as the first line of main!
    gin.parse_config_files_and_bindings(
        config_files=FLAGS.gin_configs, bindings=None, skip_unknown=True
    )

    combine_GEM_datasets(
        base_path=gin.REQUIRED,
        subjects=gin.REQUIRED,
        val_cam_idx=gin.REQUIRED,
    )

if __name__ == "__main__":    
    main()