import os 
import time
# python configs/autosubmit_final_configs.py

# src_command = 'condor/submit.py 25 configs/distillation/FINAL/fitflame_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/pca_all_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/pca_mesh_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/predict_face_features_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/regressor_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/predict_deca_INQ807.yml'
# src_command = 'condor/submit.py 25 configs/distillation/FINAL/predict_emoca_INQ807.yml'
src_command = 'condor/submit.py 25 configs/distillation/FINAL/val_preds_INQ807.yml'

src_subject = 'INQ807'
target_subjects = [
    'APP152', 
    'PGO261', 
    'TCE049', 
    'UHV563'
    ]
skip_src_command = False

all_commands = [] if skip_src_command else [src_command]
all_commands.extend([src_command.replace(src_subject, target_subject) for target_subject in target_subjects])

for c in all_commands:
    print(c)
    os.system(c)
    time.sleep(1.1)
    print()