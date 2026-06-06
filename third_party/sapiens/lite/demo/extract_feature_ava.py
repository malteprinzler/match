# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gc
import multiprocessing as mp
import os
import time
from argparse import ArgumentParser
from functools import partial
from multiprocessing import cpu_count, Pool, Process
from typing import Union
import pillow_avif
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from adhoc_image_dataset import AdhocImageDataset
from zip_image_dataset import MultiZipImageDataset
import utils
from classes_and_palettes import GOLIATH_CLASSES, GOLIATH_PALETTE
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import einops
import os
import gc

from worker_pool import WorkerPool

torchvision.disable_beta_transforms_warning()

timings = {}
BATCH_SIZE = 32


def warmup_model(model, batch_size):
    # Warm up the model with a dummy input.
    imgs = torch.randn(batch_size, 3, 1024, 768).to(dtype=torch.bfloat16).cuda()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):
        for i in range(3):
            model(imgs)
    torch.cuda.current_stream().wait_stream(s)
    imgs = imgs.detach().cpu().float().numpy()
    del imgs, s


def inference_model(model, imgs, dtype=torch.bfloat16):
    # forward the model
    with torch.no_grad():
        (results,) = model(imgs.to(dtype).cuda())
        imgs.cpu()

    return results


def fake_pad_images_to_batchsize(imgs):
    return F.pad(imgs, (0, 0, 0, 0, 0, 0, 0, BATCH_SIZE - imgs.shape[0]), value=0)


def feat_save(feature, orig_img, output_path):
    orig_img = orig_img.data.numpy() ## bgr image


    pred_save_path = os.path.join(
        output_path.replace(".jpg", ".npy")
        .replace(".jpeg", ".npy")
        .replace(".png", ".npy")
    )
    pred_save_path_tmp = pred_save_path.replace('.npy', '_tmp.npy')
    vis_save_path = pred_save_path.replace('.npy', '.jpg')
    Path(pred_save_path).parent.mkdir(exist_ok=True, parents=True)

    # getting region of interest (assuming padded image)
    orig_h, orig_w = orig_img.shape[:2]
    resize_factor = 1024 / max(orig_h, orig_w)
    resized_h = int(round(orig_h*resize_factor))
    resized_w = int(round(orig_w*resize_factor))
    feat_roi_h = int(np.ceil(resized_h / 16))
    feat_roi_w = int(np.ceil(resized_w / 16))
    pad_l = int((1024 - resized_w) / 2 // 16)
    pad_t = int((1024 - resized_h) / 2 // 16)
    feature = feature[:, pad_t:pad_t + feat_roi_h, pad_l:pad_l +feat_roi_w]

    # # making visualization
    # pca_vis = utils.pca_rgb(einops.rearrange(feature, 'c h w -> h w c'))
    # pca_vis = cv2.resize(pca_vis, (feat_roi_w*16, feat_roi_h*16), interpolation=cv2.INTER_NEAREST)
    # vis_img = np.zeros_like(pca_vis)
    # vis_img[:resized_h, :resized_w] = cv2.resize(orig_img, (resized_w, resized_h), interpolation = cv2.INTER_LINEAR)
    # vis_img = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    # vis_img = (vis_img * .2 + pca_vis * .8).astype(np.uint8)
    # Image.fromarray(vis_img).save(vis_save_path)   

    # convert to half precision to save space
    feature = feature.astype(np.float16)

    np.save(pred_save_path_tmp, feature)
    os.replace(pred_save_path_tmp, pred_save_path)
    del feature


class AsyncSaver:
    def __init__(self, max_workers=4, max_queue=32):
        self.pool = mp.Pool(processes=max_workers)
        self.sem = mp.Semaphore(max_queue)  # limit number of in-flight jobs
        self.results = []

    def submit(self, feature, orig_img, output_path):
        self.sem.acquire()  # block if too many jobs pending
        res = self.pool.apply_async(
            feat_save, (feature, orig_img, output_path),
            callback=lambda _: self.sem.release(),
            error_callback=lambda e: (print(f"❌ Save failed: {e}"), self.sem.release())
        )
        self.results.append(res)

    def close(self):
        # Wait for all jobs to finish
        for r in tqdm(self.results, desc="Waiting for async saves"):
            r.wait()
        self.pool.close()
        self.pool.join()


def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def main():
    parser = ArgumentParser()
    parser.add_argument("checkpoint", help="Checkpoint file for pose")
    parser.add_argument("--input", type=str, default="", help="Image/Video file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="root of the output img file. "
        "Default not saving the visualization images.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=4,
        help="Set batch size to do batch inference. ",
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Model inference dtype"
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs="+",
        default=[1024, 1024],
        help="input image size (height, width)",
    )
    parser.add_argument('--frame_stride', type=int, default=1)
    parser.add_argument('--cameras', default=[], type=str, nargs='+')
    args = parser.parse_args()

    assert args.output_root != ""
    assert args.input != ""

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3,) + tuple(args.shape)
    else:
        raise ValueError("invalid input shape")

    mp.log_to_stderr()
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True

    start = time.time()

    os.makedirs(args.output_root, exist_ok=True)

    USE_TORCHSCRIPT = '_torchscript' in args.checkpoint

    # build the model from a checkpoint file
    model = load_model(args.checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if args.fp16 else torch.bfloat16
        model.to(dtype)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        model = model.to(args.device)

    input = args.input

    # Check if the input is a directory or a text file
    if os.path.isfile(input) and input.endswith(".txt"):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, "r") as file:
            image_paths = [line.strip() for line in file if line.strip()]
    else:
        raise ValueError("Invalid input, must be a directory or a text file")

    if len(image_paths) == 0:
        raise ValueError("No images found in the input directory")

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    global BATCH_SIZE
    BATCH_SIZE = args.batch_size

    n_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size

    inference_dataset = MultiZipImageDataset(
        image_list=image_paths,
        shape=(input_shape[1], input_shape[2]),
        mean=[123.5, 116.5, 103.5],
        std=[58.5, 57.0, 57.5],
        cameras = args.cameras,
        frame_stride = args.frame_stride,
        pad=True,
    )
    inference_dataloader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(min(args.batch_size, cpu_count()), 1),
    )

    saver = AsyncSaver(max_workers=max(min(args.batch_size, cpu_count()), 1), max_queue=2*BATCH_SIZE)
    for batch_idx, (batch_zip_image_path, batch_orig_imgs, batch_imgs) in tqdm(
        enumerate(inference_dataloader), total=len(inference_dataloader)
    ):
        valid_images_len = len(batch_imgs)
            
        # calculating out paths
        out_paths = []
        for p in batch_zip_image_path[:valid_images_len]:
            zip_path, img_path = p.split('.zip/')
            capture_str = zip_path.split('/')[-4]
            cam_frame_str = img_path.split('.')[0]
            out_paths.append(os.path.join(args.output_root, capture_str, cam_frame_str+'.npy'))

        # skip existing files
        if all([os.path.exists(p) for p in out_paths]):
            continue

        batch_imgs = fake_pad_images_to_batchsize(batch_imgs)
        result = inference_model(model, batch_imgs, dtype=dtype)
        for r, orig_img, out_path in zip(
                result[:valid_images_len],
                batch_orig_imgs[:valid_images_len],
                out_paths[:valid_images_len],
            ):
            saver.submit(
                    r.cpu().float().numpy(),
                    orig_img,
                    out_path,
                )
        gc.collect()
    saver.close()
    total_time = time.time() - start
    fps = 1 / ((time.time() - start) / len(image_paths))
    print(
        f"\033[92mTotal inference time: {total_time:.2f} seconds. FPS: {fps:.2f}\033[0m"
    )


if __name__ == "__main__":
    main()
