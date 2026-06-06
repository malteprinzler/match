#!/bin/bash
source /home/mprinzler/miniconda3/etc/profile.d/conda.sh
conda activate tempeh
python trainer/train_global.py \
  --config-filename /home/mprinzler/projects/gintern/TEMPEH/runs/coarse/coarse__tempeh_ava256__August28__19-15-01/config.json \
  --num-iterations 800000 \
  --weight-points2surface 10.0 \
  --weight-edge-regularizer 1.0 \
  --point-mask-weights '{"w_point_face": 0.0, "w_point_ears": 0.0, "w_point_eyeballs": 1.0, "w_point_eye_region": 0.0, "w_point_lips": 0.0, "w_point_neck": 0.0, "w_point_nostrils": 0.0, "w_point_scalp": 0.0,"w_point_boundary": 0.0}' \
  --train-data-list-fname /home/mprinzler/projects/gintern/TEMPEH/data/training_data/seventy_subj__all_seq_frames_per_seq_40_head_rot_120_train.json \
  --val-data-list-fname /home/mprinzler/projects/gintern/TEMPEH/data/training_data/eight_subj__all_seq_frames_per_seq_5_val.json