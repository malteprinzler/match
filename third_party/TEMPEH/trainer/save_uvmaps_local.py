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

python trainer/save_uvmaps_local.py \
    --config-filename /home/mprinzler/projects/gintern/TEMPEH/runs/refinement/refinement__tempeh_fine_ava256_scale08_wedgeface10__September11__01-52-50/config.json \
    --outpath /fast/mprinzler/gintern/datasets/ava-256_uvpredictions/TEMPEH_ORIGINAL/refinement__tempeh_fine_ava256_scale08_wedgeface10__September11__01-52-50_debug \
    --uv-out-height 786 \
    --uv-out-width 512

"""

import os
from pathlib import Path
from option_handler.train_options_local import TrainOptions
import pudb
from argparse import ArgumentParser
from trainer.local_trainer import save_uvmaps


def get_date_string():
    from datetime import datetime
    mydate = datetime.now()
    return '%s%02d__%02d-%02d-%02d' % (mydate.strftime("%B"), mydate.day, mydate.hour, mydate.minute, mydate.second)

def execute_locally(config_fname, outpath):
    process_idx = int(os.environ.get('CONDOR_Process', '0'))
    world_size = int(os.environ.get('CONDOR_WORLD_SIZE', '1'))
    save_uvmaps(config_fname=config_fname, outpath=outpath, process_idx=process_idx, world_size=world_size)

def main():
    parser = TrainOptions()
    config_args = parser.parse()
    config_fname = config_args.config_filename
    outpath = config_args.outpath
    execute_locally(config_fname, outpath)

if __name__ == '__main__':
    main()
    print('Done')



