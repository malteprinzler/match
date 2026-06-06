#!/bin/bash
conda deactivate
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch=2.3 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks six pytz zipp