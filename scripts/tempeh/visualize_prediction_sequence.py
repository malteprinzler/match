#!/usr/bin/env python3
"""Render AVA256 coarse mesh predictions to a video."""

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import subprocess
import tempfile
from pathlib import Path

import einops
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
import tqdm
from PIL import Image

from match.data.ava256_dataset import AvaMultiCaptureDataset
from match.data.base_dataset import BaseMultiCaptureDataset
from match.utils import mesh_util, render_util
from match import data
from match.utils.data_util import MatchBatch


def _run_ffmpeg(frame_pattern: Path, fps: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def _save_frame(frame: np.ndarray, output_path: Path) -> None:
    Image.fromarray(frame).save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render AVA256 coarse mesh predictions under selected cameras."
    )
    parser.add_argument("--config", type=str, required=True, help="config file")
    return parser.parse_args()



def visualize_prediction_sequence(dataset: BaseMultiCaptureDataset, output_path: Path, contrast: float = 1.0, fps:int = 24, nframes:int = -1) -> None:
    mesh_info = mesh_util.load_obj("assets/ava256/face_topology_cleaned.obj")
    faces = mesh_info["vi"].astype(np.int32)
    output_path = Path(output_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    dataiter = iter(dataloader)
    with tempfile.TemporaryDirectory(prefix="ava256_mesh_frames_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        sample_count = len(dataset) if nframes == -1 else min(nframes, len(dataset))

        max_workers = min(8, max(1, os.cpu_count() or 1))
        max_pending = max_workers * 4
        pending: deque[Future] = deque()

        written = 0
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx in tqdm.trange(sample_count, desc="Rendering samples"):
                batch = next(dataiter)
                sample = MatchBatch(batch).squeeze()
                try:
                    mesh_renders = render_util.render_mesh_from_sample(sample=sample, faces=faces, contrast=contrast)
                except ValueError as exc:
                    print(f"Skipping sample {idx}: {exc}")
                    continue
                frame = np.concatenate([(einops.rearrange(
                    sample['image'][:5], 'v c h w -> h (v w) c').numpy()*255).astype(np.uint8),
                    einops.rearrange(mesh_renders[:5], 'v h w c -> h (v w) c')], axis=0)
                pending.append(
                    pool.submit(_save_frame, frame, tmpdir_path / f"frame_{written:06d}.jpg")
                )
                written += 1

                if len(pending) >= max_pending:
                    pending.popleft().result()

            while pending:
                pending.popleft().result()

        if written == 0:
            raise RuntimeError("No frames were rendered. Check dataset filters and coarse meshes.")

        _run_ffmpeg(
            frame_pattern=tmpdir_path / "frame_%06d.jpg",
            fps=fps,
            output_path=output_path,
        )
    print(f"Saved video to {output_path}")

def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    OmegaConf.resolve(config)
    config = config.visualize_prediction_sequence_config
    
    dataset = data.get_dataset_cls(config.dataset_cls)(
        **config.dataset_kwargs,
    )
    visualize_prediction_sequence(dataset, **config.kwargs)

    


if __name__ == "__main__":
    main()
