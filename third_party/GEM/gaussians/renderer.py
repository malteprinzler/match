# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2025 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: wojciech.zielonka@tuebingen.mpg.de, wojciech.zielonka@tu-darmstadt.de

import pudb
import torch as th
import torch
from gsplat import rasterization, rasterization_2dgs
import torch.nn.functional as F
import math
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from lib.common import parse_payload
from gaussians.cameras import batch_to_camera
from utils.image import paste
from utils.general import get_single
from collections import defaultdict
import einops


bg_colors = {
    "white": th.tensor([1, 1, 1], dtype=th.float32, device="cuda").float(),
    "black": th.tensor([0, 0, 0], dtype=th.float32, device="cuda").float(),
}

# We need to ad
bg_maps = {
    "white": None,
    "black": None,
    "random": None
}


def splat(batch, results, bg_color, to_canonical=False, twoDgs=False):
    root_RT = batch["root_RT"]

    if to_canonical:
        root_RT = None

    render_pkg = parse_payload(results, root_RT)

    pkg = render(batch, render_pkg, bg_color=bg_color, twoDgs=twoDgs)

    pred_image = pkg["render"]
    pred_alpha = pkg["alpha"]
    bg_color = pkg["bg_color"]

    return pred_image, render_pkg, pred_alpha, bg_color

def paste(img, crop):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], int(crop[4]), int(crop[5])
    if left_w > right_w:
        img = img[:, :, :W]
    else:
        img = img[:, :, -W:]
    if top_h > bottom_h:
        img = img[:, :H, :]
    else:
        img = img[:, -H:, :]

    return img


def pad_image(img, crop, h, w):
    left_w, right_w, top_h, bottom_h, W, H = crop[0], crop[1], crop[2], crop[3], crop[4], crop[5]
    left, right, up, bottom = 0, 0, 0, 0
    dx = int(abs(w - W))
    dy = int(abs(H - h))
    if left_w > right_w:
        right = dx
    else:
        left = dx
    if top_h > bottom_h:
        bottom = dy
    else:
        up = dy

    padded = F.pad(img, (left, right, up, bottom, 0, 0), "constant", 0)

    return padded

def rasterization_2dgs_wrapper(*args, **kwargs):
    '''unifies output of rasterization() and rasterization_2dgs'''
    colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(*args, **kwargs)

    meta.update(dict(normals=normals, surf_normals=surf_normals, distort=distort, median_depth=median_depth))
    return colors, alphas, meta

def render(batch, pkg, bg_color="black", twoDgs=False, training=True):
    '''
    using gsplat
    '''
    rasterization_fn = rasterization_2dgs_wrapper if twoDgs else rasterization
    is_batch = len(pkg['means3D'].shape) == 3
    if not is_batch:
        pkg = dict([(k, v.unsqueeze(0)) for k, v in pkg.items()])
        batch = dict([(k, v.unsqueeze(0)) for k, v in batch.items()])
    B, C, H, W = batch['image'].shape


    if bg_color != "random":
        bg_color_tensor = einops.repeat(bg_colors[bg_color], 'c -> b 1 c', b=B)
    else:
        bg_color_tensor = th.rand([B,1, 3]).cuda()

    K = batch['K']
    viewmat = batch['cam_RT']

    colors, sh_degree = None, None
    if "shs" in pkg:
        raise NotImplementedError('Havnt implemented active sh degree yet')
        colors = pkg_["shs"].contiguous()
    else:
        colors = pkg["colors_precomp"].contiguous()

    means3D = pkg["means3D"].contiguous()
    opacities = pkg["opacity"].contiguous()
    rotations = pkg["rotation"].contiguous()
    scales = pkg["scales"].contiguous()
    render_colors, render_alphas, info = rasterization_fn(
        means=means3D,  # [B, N, 3]
        quats=rotations,  # [B, N, 4]
        scales=scales,  # [B, N, 3]
        opacities=opacities.squeeze(-1),  # [B, N,]
        colors=colors,
        viewmats=viewmat.unsqueeze(1),  # [B, 1, 4, 4]
        Ks=K.unsqueeze(1),  # [B, 1, 3, 3]
        backgrounds=bg_color_tensor,
        width=W,
        height=H,
        packed=False,
        sh_degree=sh_degree,
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = einops.rearrange(render_colors, 'b 1 h w c -> b c h w')
    alpha = einops.rearrange(render_alphas, 'b 1 h w c -> b c h w')
    radii = einops.rearrange(info["radii"], 'b 1 n c -> b n c') # [B, N, 2]
    bg_color_tensor = einops.rearrange(bg_color_tensor, 'b 1 c -> b c')
    try:
        info["means2d"].retain_grad() # [B, N, 2]
    except:
        pass


    returns = {
        'render':rendered_image, 
        'alpha':alpha, 
        'visibility_filter':torch.all(radii > 0, dim=-1), 
        'bg_color':bg_color_tensor, 
        }

    if not is_batch:
        returns = dict([(k, v.squeeze(0)) for k, v in returns.items()])
    return returns



def render_original(batch, pkg, bg_color="black", training=True):
    '''using original diff-gaussian-rasterization'''
    viewpoint_camera = batch_to_camera(batch)

    # Set up background color
    crop = batch["crop"].cpu().numpy()

    if bg_color != "random":
        color = bg_colors[bg_color]
    else:
        color = th.randn([3]).cuda()

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pkg["sh_degree"] if "sh_degree" in pkg else 0,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    shs, colors_precomp = None, None
    if "shs" in pkg:
        shs = pkg["shs"]
    else:
        colors_precomp = pkg["colors_precomp"].contiguous()

    means3D = pkg["means3D"].contiguous()
    opacities = pkg["opacity"].contiguous()
    rotations = pkg["rotation"].contiguous()
    scales = pkg["scales"].contiguous()

    screenspace_points = th.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0

    try:
        screenspace_points.retain_grad()
    except:
        pass

    means2D = screenspace_points

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rasterizer.train(mode=training)

    rendered_image, radii, _, alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )

    # depth = paste(depth, crop)

    return {
        # "depth": depth,
        "render": paste(rendered_image, crop),
        "alpha": paste(alpha, crop),
        # "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        # "radii": radii,
        "bg_color": color
    }
