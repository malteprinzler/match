from pathlib import Path
import pudb
# python configs/autocreate_final_configs.py

src_subject = 'INQ807'
next_subject = 'APP152'
target_subjects = ['APP152', 'PGO261', 'TCE049', 'UHV563']
src_paths = [
    Path('configs/distillation/FINAL/fitflame_INQ807.yml'),
    Path('configs/distillation/FINAL/pca_all_INQ807.yml'),
    Path('configs/distillation/FINAL/pca_mesh_INQ807.yml'),
    Path('configs/distillation/FINAL/predict_face_features_INQ807.yml'),
    Path('configs/distillation/FINAL/regressor_INQ807.yml'),
    Path('configs/distillation/FINAL/predict_deca_INQ807.yml'),
    Path('configs/distillation/FINAL/predict_emoca_INQ807.yml'),
    Path('configs/distillation/FINAL/val_preds_INQ807.yml'),
]

all_subjects = [src_subject] + target_subjects

target_subject_token = '__TARGET_SUBJECT__'
next_subject_token = '__NEXT_SUBJECT__'

for sp in src_paths:
    with sp.open('r') as f:
        content = f.read()
    for i, ts in enumerate(target_subjects):
        next_subject_ = all_subjects[(i+2)%len(all_subjects)]
        content_ = content.replace(src_subject, target_subject_token)
        content_ = content_.replace(next_subject, next_subject_token)
        content_ = content_.replace(target_subject_token, ts)
        content_ = content_.replace(next_subject_token, next_subject_)
        tp = Path(str(sp).replace(src_subject, ts))
        tp.parent.mkdir(exist_ok=True, parents=True)
        with tp.open('w') as f:
            f.write(content_)
        print(f'Created {tp}')

