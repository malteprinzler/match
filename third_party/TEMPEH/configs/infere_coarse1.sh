#!/bin/bash
source /home/mprinzler/miniconda3/etc/profile.d/conda.sh
conda activate tempeh
python tester/test_global.py \
                  --coarse_model_run_dir '/is/cluster/mprinzler/projects/gintern/TEMPEH/runs/coarse/coarse__tempeh_orig__February13__10-01-07_coarse1' \
                  --data_list_fname './data/training_data/seventyeight_subj_trainval.json' \
                  --image_directory './data/training_data/downsampled_images_4' \
                  --calibration_directory './data/training_data/calibrations' \
                  --out_dir './runs/coarse/coarse__tempeh_orig__February13__10-01-07_coarse1/train_inference'
