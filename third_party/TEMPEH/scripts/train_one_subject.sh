#!/bin/bash
source /home/mprinzler/miniconda3/etc/profile.d/conda.sh
conda activate tempeh
python trainer/train_global.py \
  --num-iterations 300000 \
  --train-data-list-fname /is/cluster/mprinzler/projects/gintern/TEMPEH/data/training_data/one_subj__all_seq_frames_per_seq_40_head_rot_120_train.json \
  --val-data-list-fname /is/cluster/mprinzler/projects/gintern/TEMPEH/data/training_data/one_subj__all_seq_frames_per_seq_5_val.json
