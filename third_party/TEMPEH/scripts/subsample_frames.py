import json
from pathlib import Path
import numpy as np

in_path = Path('/is/cluster/mprinzler/projects/gintern/TEMPEH/data/training_data/one_subj__all_seq_frames_per_seq_40_head_rot_120_train.json')
out_path = Path('/is/cluster/mprinzler/projects/gintern/TEMPEH/data/training_data/one_subj__all_seq_frames_per_seq_5_val.json')
nframes_per_seq = 5

# reading in frames
frame_dict = dict()
with open(str(in_path), 'r') as f:
    in_frames = json.load(f)
for subject, sequence, frame in in_frames:
    if not subject in frame_dict:
        frame_dict[subject] = dict()
    if not sequence in frame_dict[subject]:
        frame_dict[subject][sequence] = list()
    frame_dict[subject][sequence].append(frame)


# getting out frames
out_frames = list()
for subject in sorted(frame_dict.keys()):
    for sequence in sorted(frame_dict[subject].keys()):
        frame_dict[subject][sequence] = sorted(frame_dict[subject][sequence])
        n_frames = len(frame_dict[subject][sequence])
        sel_fidcs = np.round(np.linspace(0, n_frames-1, min(nframes_per_seq, n_frames))).astype(int)
        for fidx in sel_fidcs:
            out_frames.append([subject, sequence, frame_dict[subject][sequence][fidx]])

# writing out frames
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(out_frames, f, indent='\t')
print(f'Written {len(out_frames)} frames to {out_path}')