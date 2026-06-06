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

import trimesh
import copy
import einops
import pudb
from pathlib import Path
import sys
import numpy as np
import torch as th
import torch.utils.data
from sklearn.decomposition import PCA
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
from data.base import DatasetMode
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_raw_multiply, standardize_quaternion
from lib.apperance.trainer import ApperanceTrainer
from train import folders
from lib.F3DMM.masks.masking import Masking
from utils.general import build_dataset, build_loader, get_single, seed_everything, to_device
from pca_viewer import splat_coeffs
from lib.common import Mesh, add_joint, rigid_trafo_gaussians, align_quaternions, average_gaussians
from lib.common import vis_3d_point_clouds, invert_c2w
from torch import Tensor
from gaussians.renderer import splat
from lib.common import add_joint, remove_joint, make_vis_gaussians
from typing import Union
from lib.apperance.pca_gaussian import PCApperance
from torchvision.utils import save_image
from utils.geometry import AttrDict
torch.backends.cudnn.benchmark = True
from pca_viewer import save_video
import shutil
import os
from typing import Union, TextIO, List
import pudb
from lib.fit_flame import fit_flame_to_flame_vertices
from lib.F3DMM.FLAME2023.flame import FLAME
from collections import defaultdict
from lib.common import unpose_gaussians
from lib.common import pose_gaussians
from lib.common import convert_flame_fits_to_dataset_flame_params
from utils.renderer import Renderer
renderer = Renderer(white_background=True).cuda()




def load_obj(path: Union[str, TextIO], return_vn: bool = False):
    """Load wavefront OBJ from file. See https://en.wikipedia.org/wiki/Wavefront_.obj_file for file format details
    Args:
        path: Where to load the obj file from
        return_vn: Whether we should return vertex normals

    Returns:
        Dictionary with the following entries
            v: n-by-3 float32 numpy array of vertices in x,y,z format
            vt: n-by-2 float32 numpy array of texture coordinates in uv format
            vi: n-by-3 int32 numpy array of vertex indices into `v`, each defining a face.
            vti: n-by-3 int32 numpy array of vertex texture indices into `vt`, each defining a face
            vn: (if requested) n-by-3 numpy array of normals
    """

    if isinstance(path, str):
        with open(path, "r") as f:
            lines: List[str] = f.readlines()
    else:
        lines: List[str] = path.readlines()

    v = []
    vt = []
    vindices = []
    vtindices = []
    vn = []

    for line in lines:
        if line == "":
            break

        if line[:2] == "v ":
            v.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vt":
            vt.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "vn":
            vn.append([float(x) for x in line.split()[1:]])
        elif line[:2] == "f ":
            vindices.append([int(entry.split("/")[0]) - 1 for entry in line.split()[1:]])
            if line.find("/") != -1:
                vtindices.append([int(entry.split("/")[1]) - 1 for entry in line.split()[1:]])

    if len(vt) == 0:
        assert len(vtindices) == 0, "Tried to load an OBJ with texcoord indices but no texcoords!"
        vt = [[0.5, 0.5]]
        vtindices = [[0, 0, 0]] * len(vindices)

    # If we have mixed face types (tris/quads/etc...), we can't create a
    # non-ragged array for vi / vti.
    mixed_faces = False
    for vi in vindices:
        if len(vi) != len(vindices[0]):
            mixed_faces = True
            break

    if mixed_faces:
        vi = [np.array(vi, dtype=np.int32) for vi in vindices]
        vti = [np.array(vti, dtype=np.int32) for vti in vtindices]
    else:
        vi = np.array(vindices, dtype=np.int32)
        vti = np.array(vtindices, dtype=np.int32)

    out = {
        "v": np.array(v, dtype=np.float32),
        "vn": np.array(vn, dtype=np.float32),
        "vt": np.array(vt, dtype=np.float32),
        "vi": vi,
        "vti": vti,
    }

    if return_vn:
        assert len(out["vn"]) > 0
        return out
    else:
        out.pop("vn")
        return out






def to_fp16(obj):
    if isinstance(obj, dict):
        return {key: to_fp16(val) for key, val in obj.items()}
    elif isinstance(obj, list):
        return [to_fp16(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.astype(np.float16)
    else:
        return obj


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    rot_mats = quaternion_to_matrix(quaternions)
    axis_angles = matrix_to_axis_angle(rot_mats)
    return axis_angles


def merge_textures(pkg, single, tex_to_mesh):
    if "pred" in pkg:
        pkg = pkg.pred

    geometry = pkg.means3D

    geometry = remove_joint(geometry, single["A"], single["W"], tex_to_mesh, joint=1)

    payload = {
        "geometry": geometry,
        "opacity": pkg.opacity,
        "scales": pkg.scales,
        # "rotation": quaternion_to_axis_angle(pkg.rotation),
        "rotation": pkg.rotation,
        "colors": pkg.colors_precomp
    }

    return payload


def save_xyz(filename, pc):
    pc = pc.cpu().numpy()
    np.savetxt(filename, pc, fmt='%.6f', delimiter=' ', header='x y z', comments='')

GTEMPEH_TO_GEM_TRAFO = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

def get_offseted_gtempeh_to_gem_trafo(origin_offset=None):
    """inverting y and z axis and applying origin offset
    
    Returns: 
        trafo of shape (4,4)
    """
    trafo = GTEMPEH_TO_GEM_TRAFO
    if origin_offset is not None:
        offset_trafo = torch.diag(torch.tensor([1., 1., 1., 1.]).to(trafo))
        offset_trafo[:3, -1] = -origin_offset.to(trafo)
        trafo = offset_trafo @ trafo 
    return trafo


def gtempeh_gaussians_2_gem_gaussians(gaussians:dict, origin_offset=None):
    gaussians = copy.deepcopy(gaussians)
    mask = gaussians['mask'][0,0]  # (Huv, Wuv)
    # masking the gaussians 
    for k, v in gaussians.items():
        gaussians[k] = einops.rearrange(v[0], 'c h w -> h w c')[mask]
    
    trafo = get_offseted_gtempeh_to_gem_trafo(origin_offset=origin_offset)

    gem_gaussians = AttrDict(dict(
        geometry=gaussians['xyz'], 
        opacity=gaussians['opacity'],
        scales=gaussians['scale'],
        rotation=gaussians['rotation'],
        colors=gaussians['rgb'])
    )
    gem_gaussians = rigid_trafo_gaussians(gem_gaussians, trafo, ref_quaternion=standardize_quaternion(matrix_to_quaternion(trafo[:3, :3])))
    return gem_gaussians


from torchvision.transforms.functional import resize
def resize_batch(batch, H, W):
    batch = copy.deepcopy(batch)
    h, w = batch['image'].shape[-2:]
    for k in ['image', 'alpha', 'img_loss_mask']:
        batch[k] = resize(batch[k], [H, W])
    batch['K'][..., 0, :] *= W/w
    batch['K'][..., 1, :] *= H/h
    return batch


def gtempeh_points_2_gem_points(points: torch.Tensor, origin_offset=None):
    '''
    Args:
        points: (N, 3)
    '''
    
    trafo = get_offseted_gtempeh_to_gem_trafo(origin_offset=origin_offset).to(points)
    points = torch.cat((points, torch.ones_like(points[:,:1])), dim=-1)
    points = einops.einsum(trafo, points, 'i j, n j -> n i')
    points = points[:, :3]
    return points



def get_gtempeh_origin_offset(gtempeh_cameras, single):
    
    cam_idx = single['cam']
    if isinstance(cam_idx, torch.Tensor):
        cam_idx = cam_idx.cpu().item()
    camcenter = invert_c2w(single['cam_RT'])[:3, -1]
    if isinstance(camcenter, np.ndarray):
        camcenter = torch.from_numpy(camcenter).to(gtempeh_cameras)
    gtempeh_camcenter = gtempeh_cameras[cam_idx, :3, -1]
    gtempeh_camcenter = einops.einsum(GTEMPEH_TO_GEM_TRAFO.to(gtempeh_camcenter), 
                                      torch.cat([gtempeh_camcenter, torch.ones_like(gtempeh_camcenter[:1])], 
                                                dim=-1),
                                                'i j, j -> i')[:3]
    gtempeh_origin_offset = gtempeh_camcenter - camcenter
    return gtempeh_origin_offset

def get_pc_to_mesh(pc, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices.cpu(), faces=faces.cpu(), process=False)
    pc_to_mesh = mesh.kdtree.query(pc.cpu())[1]
    return th.from_numpy(pc_to_mesh).to(pc.device)


def gaussian_locations_2_mesh(xyz, triangle_uvs, triangle_vert_ids, num_verts):
    """Converts gaussian parameters to a mesh.

    Args:
      xyz: gaussian locations (B, 1, 3, H, W)

    Returns:
      verts: A torch tensor of shape (B, Verts, 3)
    """
    b = len(xyz)

    xyz = einops.rearrange(xyz, "b 1 c h w -> b c h w")
    triangle_vert_ids = triangle_vert_ids.to(xyz.device)
    triangle_uvs = (
        triangle_uvs.unsqueeze(0)
        .expand(b, -1, -1, -1)
        .to(xyz)
    )  # (B, N_faces, 3, 2)
    triangle_uvs = triangle_uvs * 2 - 1  # [0, 1] -> [-1, 1]
    triangle_uvs[..., 1] *= -1  # flip y axis: origin at bottom left -> top left
    xyz_triangles = torch.nn.functional.grid_sample(
        xyz,
        triangle_uvs,
        mode="bilinear",
        align_corners=True,  # (-1, -1) refers to center of top-left pixel
        padding_mode="border",
    )  # (B, 3(xyz), N_faces, 3(verts per face))

    # aggregating mesh vertex 3D coordinates for occurences in different triangles
    xyz_triangles = einops.rearrange(xyz_triangles, "b c f v -> b f v c")
    xyz_vert = triangle_values_to_vert_values(xyz_triangles, triangle_vert_ids=triangle_vert_ids, v=num_verts)
    return xyz_vert

def triangle_values_to_vert_values(
      triangle_values, triangle_vert_ids, v
  ):
    """Aggregates triangle values to vertex values (averaging over duplicate vertex occurences).

    Args:
      triangle_values: A torch tensor of shape (B, N_faces, 3(verts per face),C)
      triangle_vert_ids: A torch tensor of shape (N_faces, 3(verts per face))
        with vertex ids for each triangle vertex
      v: Number of vertices in the mesh. If None, it is set to the number of
        vertices in the G-Nome mesh.

    Returns:
      vert_values: A torch tensor of shape (B, V, C)
    """
    b = len(triangle_values)
    c = triangle_values.shape[-1]

    triangle_values = einops.rearrange(triangle_values, "b f v c -> b (f v) c")
    triangle_vert_ids = triangle_vert_ids.to(torch.int64)
    triangle_vert_ids = einops.rearrange(triangle_vert_ids, "f v -> 1 (f v) 1")
    triangle_vert_ids = triangle_vert_ids.expand(
        b, -1, c
    )  # (B, faces * 3vertidsperface, C))

    vert_values = torch.zeros(
        (b, v, c), dtype=triangle_values.dtype, device=triangle_values.device
    )
    vert_values.scatter_reduce_(
        dim=1,
        index=triangle_vert_ids,
        src=triangle_values,
        reduce="mean",
        include_self=False,
    )
    return vert_values


def ava2flame_verts(ava_verts, ava2flame_mapping):
    tri_vids = ava2flame_mapping['vertex_indices']
    bary = ava2flame_mapping['barycentric_coordinates'].to(ava_verts.dtype)
    flame_verts = einops.einsum(bary, ava_verts[tri_vids], 'v i, v i c -> v c')
    return flame_verts

# loading template mesh
template_path = '/home/mprinzler/projects/gintern/gtempeh/assets/ava256/face_topology_cleaned.obj'
mesh_info = load_obj(str(template_path))
template_verts = torch.from_numpy(mesh_info['v']).cuda()
template_triangles = torch.from_numpy(mesh_info['vi']).cuda()
template_triangle_uvs = torch.from_numpy(mesh_info['vt'][mesh_info['vti']]).cuda()


def draw_slider_above_image(
    img: torch.Tensor,
    sigma: float,
    slider_height=20,
    padding=10,
    line_color=1.0,
    handle_color=1.0,
    line_thickness=1,
    handle_thickness=2
):
    """
    img: (3, H, W) float tensor in [0,1]
    sigma: float in [0,1]
    slider_height: total available slider area height
    padding: vertical space between slider and image
    line_color: grayscale value for slider line
    handle_color: grayscale value for handle
    line_thickness: thickness (in px) of the slider line
    handle_thickness: thickness (in px) of the handle bar
    """

    C, H, W = img.shape
    out_H = slider_height + padding + H

    # white background
    out = torch.ones((C, out_H, W), dtype=img.dtype)

    # --- Slider line ---
    line_y_center = slider_height // 2
    half_line = line_thickness // 2
    top_line_y = max(line_y_center - half_line, 0)
    bottom_line_y = min(line_y_center + half_line + 1, slider_height)

    out[:, top_line_y:bottom_line_y, :] = line_color

    # --- Handle ---
    x_center = int(sigma * (W - 1))
    half_handle = handle_thickness // 2

    handle_left = max(x_center - half_handle, 0)
    handle_right = min(x_center + half_handle + 1, W)

    handle_top = max(2, line_y_center - slider_height // 2 + 2)
    handle_bottom = min(slider_height - 2, line_y_center + slider_height // 2 - 2)

    out[:, handle_top:handle_bottom, handle_left:handle_right] = handle_color

    # --- Paste image below slider ---
    out[:, slider_height + padding : slider_height + padding + H, :] = img

    return out

def render_mesh(cameras, mesh, mask=None, bg_color=None):
    with th.no_grad():
        vertices = mesh.v.float()
        faces = mesh.f.long()[None]
        mesh_rendering = renderer(cameras, vertices[None], faces).permute(2, 0, 1)
        if mask is not None:
            alpha = renderer.resterize_attributes(cameras, vertices[None], faces, mask)[0][0, :, :, 0].permute(2, 0, 1)
            if bg_color is None:
                bg = th.ones_like(mesh_rendering)
            else:
                bg = th.ones_like(mesh_rendering) * bg_color
            mesh_rendering = mesh_rendering * alpha + (1 - alpha) * bg
        return mesh_rendering


def run(config, quantize=False, debug_frames=None):
    use_parts = config.train.use_parts
    config.data.join_configs = True
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.validation)
    loader = build_loader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        persistent_workers=True,
        seed=33,
    )

    do_color_pca = config.train.get('dynamic_colors', False)
    use_gtempeh_predictions = config.train.get('use_gtempeh_predictions', False)
    twoDgs = config.train.get('twoDgs', False)
    if use_gtempeh_predictions: 
        assert do_color_pca

    if use_gtempeh_predictions:
        trainer = None
    else:
        trainer = ApperanceTrainer(config, dataset)
        trainer.restore()
        trainer.eval()

        # Disable Gaussian masking
        trainer.model.gaussian_mask = None
    
    masking = Masking()

    list_masks = {"full": masking.full().cuda()}
    if use_gtempeh_predictions:
        list_masks['full'] = torch.ones_like(list_masks['full'])
    if use_parts:    
        list_masks = masking.list_masks()

    uv_size = config.train.uv_size
    tex_to_mesh = None

    Path(f"checkpoints/GAUSSIAN_PCA_{uv_size}").mkdir(exist_ok=True, parents=True)
    logger.info(f"Building PCA with total of {len(dataset)} frames")

    flame = FLAME().cuda()

    rgb = None
    gaussians = {}
    original_gaussians_renders = []
    original_gaussians_vis_renders = []
    flame_params = defaultdict(list)
    ref_quaternion = None

    ava2flame_mapping = dict(np.load('/home/mprinzler/projects/gintern/GEM/assets/meshes/mapping_ava2flame.npz'))
    for k, v in ava2flame_mapping.items():
        ava2flame_mapping[k] = torch.from_numpy(v).cuda()

    flame2gaussian_mapping = dict(np.load('/home/mprinzler/projects/gintern/GEM/assets/meshes/mapping_flame2gaussians.npz'))
    for k, v in flame2gaussian_mapping.items():
        flame2gaussian_mapping[k] = torch.from_numpy(v).cuda()

    # getting tex2mesh
    if use_gtempeh_predictions:
        # getting dummy gaussians to know masking
        single = dataset[0]
        frame_idx = int(Path(single['image_path']).name.split('_')[0])
        gtempeh_gaussians_path = sorted([p for p in Path(f'{config.train.gtempeh_path}/{single["exp"]}').iterdir() if p.name.isnumeric()])[frame_idx] / 'gaussians.pt'
        gtempeh_gaussians = torch.load(gtempeh_gaussians_path)
        mask = gtempeh_gaussians['mask']
        bary_masked = einops.rearrange(flame2gaussian_mapping['barycentric_coordinates'], '1 c h w -> h w c')[mask[0,0]]  # (N, 3)
        vertid_masked = einops.rearrange(flame2gaussian_mapping['vertex_indices'], '1 c h w -> h w c')[mask[0,0]]  # (N, 3)
        biggest_bary_idx = torch.argmax(bary_masked, dim=-1, keepdim=True)
        tex_to_mesh = torch.gather(vertid_masked, dim=1, index=biggest_bary_idx)[:, 0]
    else:
        tex_to_mesh = trainer.model.tex_to_mesh


    # visualize pca components
    pca = PCApperance(config).cuda()
    
    # # disabling dynamic mouth colors
    # apperance = pca
    # pca_color_components = apperance.unzip_all(dict(all=apperance.components['all_full'].view(apperance.n, -1, apperance.channels['all_full'])))['colors']  # (ncomponents, ngaussians, 3)
    # pca_color_components[:,masking.mouth()[apperance.tex_map_full.cpu()]] = 0
    pca_dir = Path(PCApperance.get_pca_path(config)).parent
    pca_seq_dir = pca_dir/'coeff_vis_zoomedout_handle'
    pca_seq_dir.mkdir(exist_ok=True, parents=True)
    batch = next(iter(loader))
    batch = to_device(batch)
    batch = resize_batch(batch, 1280, 1024)
    batch['K'][..., 0, 0] *= 0.75
    batch['K'][..., 1, 1] *= 0.75

    counter = 0
    with th.no_grad():
        for coeff_idx in range(5):
            for coeff_value in np.sin(np.linspace(0, 2*np.pi, 48))*3:
                coeffs = torch.zeros((1, 150), device='cuda', dtype=torch.float32)
                coeffs[0, coeff_idx] = coeff_value
                coeffs = dict(all_full=coeffs)
                results = AttrDict(pca.inverse_transform(coeffs))
                flame_params = convert_flame_fits_to_dataset_flame_params(batch['flame_params'], inverse=True)
                results_posed = AttrDict(pose_gaussians(results, flame_params, flame, tex_to_mesh))
                pred_image, render_pkg, pred_alpha, bg_color = splat(batch, results_posed, bg_color='white', twoDgs=twoDgs, to_canonical=True)
                
                # add slider
                slider_kwargs = dict(handle_color=0., slider_height=40, padding=0,
                            line_color=0.0,
                            line_thickness=2, handle_thickness=16)
                for coeff_idx_ in list(range(5))[::-1]:
                    pred_image = draw_slider_above_image(pred_image[0], coeffs['all_full'][0,coeff_idx_].cpu().item()/3/2+.5, **slider_kwargs)[None]


                img_path = pca_seq_dir / f'{counter:06d}.jpg'
                save_image(pred_image[0], img_path)
                counter += 1

    save_video(str(pca_seq_dir), suffix='jpg')
    # shutil.rmtree(pca_seq_dir)

if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()
    folders(config)

    quantize = False
    debug_frames = None
    run(config, quantize, debug_frames=debug_frames)
