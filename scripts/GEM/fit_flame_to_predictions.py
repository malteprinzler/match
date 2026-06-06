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

import json
import trimesh
import copy
import einops
import pudb
from pathlib import Path
import sys
import numpy as np
import torch as th
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import torch
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, standardize_quaternion
from typing import Union
from torchvision.utils import save_image
torch.backends.cudnn.benchmark = True
import shutil
import ffmpeg
from typing import Union, TextIO, List
from torch.utils.data import default_collate

sys.path.append('third_party/GEM')
from data.base import DatasetMode
from lib.F3DMM.masks.masking import Masking
from utils.general import build_dataset, build_loader, get_single, none_collate, seed_everything, to_device
from gaussians.renderer import splat
from lib.common import Mesh, rigid_trafo_gaussians, invert_c2w, remove_joint, make_vis_gaussians
from utils.geometry import AttrDict
from lib.fit_flame import fit_flame_to_flame_vertices
from lib.F3DMM.FLAME2023.flame import FLAME
from utils.renderer import Renderer
from lib.common import unpose_gaussians, unpose_vertices, convert_flame_fits_to_dataset_flame_params
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

MATCH_TO_GEM_TRAFO = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

def get_offseted_match_to_gem_trafo(origin_offset=None):
    """inverting y and z axis and applying origin offset
    
    Returns: 
        trafo of shape (4,4)
    """
    trafo = MATCH_TO_GEM_TRAFO
    if origin_offset is not None:
        offset_trafo = torch.diag(torch.tensor([1., 1., 1., 1.]).to(trafo))
        offset_trafo[:3, -1] = -origin_offset.to(trafo)
        trafo = offset_trafo @ trafo 
    return trafo


def match_gaussians_2_gem_gaussians(gaussians:dict, origin_offset=None):
    gaussians = copy.deepcopy(gaussians)
    mask = gaussians['mask'][0,0]  # (Huv, Wuv)
    # masking the gaussians 
    for k, v in gaussians.items():
        gaussians[k] = einops.rearrange(v[0], 'c h w -> h w c')[mask]
    
    trafo = get_offseted_match_to_gem_trafo(origin_offset=origin_offset)

    gem_gaussians = AttrDict(dict(
        geometry=gaussians['xyz'], 
        opacity=gaussians['opacity'],
        scales=gaussians['scale'],
        rotation=gaussians['rotation'],
        colors=gaussians['rgb'])
    )
    gem_gaussians = rigid_trafo_gaussians(gem_gaussians, trafo, ref_quaternion=standardize_quaternion(matrix_to_quaternion(trafo[:3, :3])))
    return gem_gaussians

def match_points_2_gem_points(points: torch.Tensor, origin_offset=None):
    '''
    Args:
        points: (N, 3)
    '''
    
    trafo = get_offseted_match_to_gem_trafo(origin_offset=origin_offset).to(points)
    points = torch.cat((points, torch.ones_like(points[:,:1])), dim=-1)
    points = einops.einsum(trafo, points, 'i j, n j -> n i')
    points = points[:, :3]
    return points



def get_match_origin_offset(match_cameras, single):
    
    cam_idx = single['cam']
    if isinstance(cam_idx, torch.Tensor):
        cam_idx = cam_idx.cpu().item()
    camcenter = invert_c2w(single['cam_RT'])[:3, -1]
    if isinstance(camcenter, np.ndarray):
        camcenter = torch.from_numpy(camcenter).to(match_cameras)
    match_camcenter = match_cameras[cam_idx, :3, -1]
    match_camcenter = einops.einsum(MATCH_TO_GEM_TRAFO.to(match_camcenter), 
                                      torch.cat([match_camcenter, torch.ones_like(match_camcenter[:1])], 
                                                dim=-1),
                                                'i j, j -> i')[:3]
    match_origin_offset = match_camcenter - camcenter
    return match_origin_offset

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
template_path = 'assets/ava256/face_topology_cleaned.obj'
mesh_info = load_obj(str(template_path))
template_verts = torch.from_numpy(mesh_info['v']).cuda()
template_triangles = torch.from_numpy(mesh_info['vi']).cuda()
template_triangle_uvs = torch.from_numpy(mesh_info['vt'][mesh_info['vti']]).cuda()



@th.no_grad()
def render_mesh(batch, mesh, mask=None, bg_color=None):
    '''
    Returns:
        rendered mesh image (B, 3, H, W)
    '''
    B = len(batch['root_RT'])
    cameras = Renderer.to_cameras(batch)        
    
    Rt = batch["root_RT"]
    R = Rt[:, :3, :3]
    T = Rt[:, :3, 3]
    vertices = einops.einsum(R, mesh.v.float(), 'b i j, b n j -> b n i') + einops.rearrange(T, 'b c -> b 1 c')
    faces = einops.repeat(mesh.f.long(), 'n c -> b n c', b=B)
    mesh_rendering = renderer(cameras, vertices, faces)
    if mask is not None:
        alpha = renderer.resterize_attributes(cameras, vertices, faces, mask)[0]
        if bg_color is None:
            bg = th.zeros_like(mesh_rendering) if bg_color == "black" else th.ones_like(mesh_rendering)
        else:
            bg = th.ones_like(mesh_rendering) * bg_color
        mesh_rendering = mesh_rendering * alpha + (1 - alpha) * bg
    
    return mesh_rendering

def run(config, quantize=False, debug_frames=None):
    config.data.join_configs = False    
    datasets = [
        build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.validation),
        build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.test),]
    
    loaders = [build_loader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        persistent_workers=True,
        seed=33,
    ) for dataset in datasets]

    twoDgs = config.train.get('twoDgs', False)
    
    masking = Masking()

    flame = FLAME().cuda()

    singles = []
    gaussians = {}
    original_gaussians_renders = []
    original_gaussians_vis_renders = []
    flame_verts = []
    ref_quaternion = None

    ava2flame_mapping = dict(np.load('assets/flame/mapping_ava2flame.npz'))
    for k, v in ava2flame_mapping.items():
        ava2flame_mapping[k] = torch.from_numpy(v).cuda()

    flame2gaussian_mapping = dict(np.load('assets/flame/mapping_flame2gaussians.npz'))
    for k, v in flame2gaussian_mapping.items():
        flame2gaussian_mapping[k] = torch.from_numpy(v).cuda()

    # getting tex2mesh
    # getting dummy gaussians to know masking
    single = datasets[0][0]
    match_gaussians_path = datasets[0].gaussians_path_from_image_path(Path(single['image_path']))
    match_gaussians = torch.load(match_gaussians_path)
    mask = match_gaussians['mask']
    bary_masked = einops.rearrange(flame2gaussian_mapping['barycentric_coordinates'], '1 c h w -> h w c')[mask[0,0]]  # (N, 3)
    vertid_masked = einops.rearrange(flame2gaussian_mapping['vertex_indices'], '1 c h w -> h w c')[mask[0,0]]  # (N, 3)
    biggest_bary_idx = torch.argmax(bary_masked, dim=-1, keepdim=True)
    tex_to_mesh = torch.gather(vertid_masked, dim=1, index=biggest_bary_idx)[:, 0]


    # 1) Build Gaussian dataset

    j = 0
    with th.no_grad():
        for loader in loaders:
            for batch in tqdm(loader, desc='Loading predicted gaussians'):
                batch = to_device(batch)
                single = get_single(batch, 0)
                singles.append(single)

                match_gaussians_path = datasets[0].gaussians_path_from_image_path(Path(single['image_path']))
                match_gaussians = torch.load(match_gaussians_path)
                ava_verts_ = gaussian_locations_2_mesh(match_gaussians['xyz'][None], triangle_uvs=template_triangle_uvs, triangle_vert_ids=template_triangles, num_verts=len(template_verts))[0]
                ava_verts_ = match_points_2_gem_points(ava_verts_)
                flame_verts_ = ava2flame_verts(ava_verts=ava_verts_, ava2flame_mapping=ava2flame_mapping)
                flame_verts.append(flame_verts_)

                
                # # vis tracked flame verts vs match predictions of flame verts
                # gt_flame_verts = single['geom_vertices']
                # gt_flame_verts = einops.einsum(single['root_RT'].float(), torch.cat([gt_flame_verts.float(), torch.ones_like(single['geom_vertices'].float()[:, :1])], dim=-1), 'i j, n j -> n i')[:,:3]
                # vis_3d_point_clouds(dict(flame_verts_pred=flame_verts_.cpu(), flame_verts_gt=gt_flame_verts.cpu()), 'demos/flame_verts_pred_vs_gt.html')
                # pudb.set_trace()

                match_gaussians = match_gaussians_2_gem_gaussians(match_gaussians)
                match_gaussians_batch = AttrDict(dict([(k, v.unsqueeze(0)) for k, v in match_gaussians.items()]))
                pred_image, render_pkg, pred_alpha, bg_color = splat(batch, match_gaussians_batch, bg_color='white', twoDgs=twoDgs, to_canonical=True)
                pred_image_vis, render_pkg, pred_alpha, bg_color = splat(batch, make_vis_gaussians(match_gaussians_batch), bg_color='white', twoDgs=twoDgs, to_canonical=True)
                original_gaussians_renders.append(pred_image.squeeze(0).cpu())
                original_gaussians_vis_renders.append(pred_image_vis.squeeze(0).cpu())
                # # visualizing point clouds (watch out, if enabled have to fix tex_to_mesh since match will use GEM tex2mesh)
                # verts = einops.einsum(single['root_RT'].float(), torch.cat([single['geom_vertices'].float(), torch.ones_like(single['geom_vertices'].float()[:, :1])], dim=-1), 'i j, n j -> n i')[:,:3]
                # vis_3d_point_clouds(dict(match_pc=match_gaussians['geometry'].cpu().numpy(), verts=verts.cpu().numpy()), 'pc_comparison.html')
                # pudb.set_trace()

                # inverse_root_RT = invert_c2w(single['root_RT'])
                # gaussians_origaligned_ = copy.deepcopy(match_gaussians)
                # if ref_quaternion is None:
                #     ref_quaternion = standardize_quaternion(matrix_to_quaternion(inverse_root_RT[:3,:3]))
                # gaussians_origaligned_ = rigid_trafo_gaussians(gaussians_origaligned_, inverse_root_RT, ref_quaternion=ref_quaternion)  # to canonical space
                # if tex_to_mesh_orig is None:
                #     tex_to_mesh_orig = get_pc_to_mesh(gaussians_origaligned_['geometry'], vertices=single['geom_vertices'], faces=single['geom_faces'])
                # gaussians_origaligned_['geometry'] = remove_joint(gaussians_origaligned_['geometry'], single["A"], single["W"], tex_to_mesh_orig, joint=1)
                # for k, v in gaussians_origaligned_.items():
                #     gaussians_origaligned[k].append(v)


                merged_match = match_gaussians
                # # visualizing point clouds (watch out, if enabled have to fix tex_to_mesh since match will use GEM tex2mesh)
                # vis_3d_point_clouds(dict(GEM_pc=merged['geometry'].cpu().numpy(), match_pc=merged_match['geometry'].cpu().numpy()), 'pc_comparison.html')
                # pudb.set_trace()
                merged=merged_match

                for k in merged.keys():
                    if k not in gaussians:
                        gaussians[k] = []
                    gaussians[k].append(merged[k].cpu())

                j += 1
                if (debug_frames is not None) and j == debug_frames: 
                    break
            if (debug_frames is not None) and j == debug_frames: 
                break

    for k,v in gaussians.items():
        gaussians[k] = torch.stack(v)

    # # visualizing masks
    # part_xyz = dict()
    # # check if list_masks cover everything:
    # full_mask = torch.zeros((5023,), dtype=torch.bool)
    # for part_mask in masking.list_masks().values():
    #     full_mask[part_mask] = True
    # assert torch.all(full_mask) 
    # for mask_name, part_mask in masking.list_masks().items():
    #     xyz = gaussians['geometry'][0]
    #     mask = part_mask[tex_to_mesh].to(xyz.device)
    #     xyz = xyz[mask]
    #     part_xyz[mask_name] = xyz.cpu().numpy()
    # vis_3d_point_clouds(part_xyz, 'demos/part_xyz.html')
    # pudb.set_trace()

    # fitting flame
    flame_verts = torch.stack(flame_verts)
    mask = ava2flame_mapping['distances']<0.005
    mask[masking.eyeballs()] = False
    mask[masking.eye_region()] = False
    # mask[masking.neck()] = False  
    mask[masking._masks.scalp] = False  # entire scalp including neck&scalp overlap

    flame_fits = fit_flame_to_flame_vertices(target_vertices=flame_verts,
                                            mask=mask,
                                            static_eyes=True,
                                            static_eyelids=True)
    flame_vertices_fit = flame(**flame_fits)[0]

    # # 3d visualization of gaussian points, flame target points and flame fit points 
    # idx = 0
    # vis_3d_point_clouds(dict(gaussians=gaussians['geometry'][idx, ::10].cpu().numpy(), flame_target=flame_verts[idx].cpu().numpy(), flame_fit = flame_vertices_fit[idx].cpu().numpy()), 'demos/flame_fit_3d.html')
    
    # saving results
    for i in tqdm(list(range(len(singles))), desc='Saving flame fits'):
        flame_track_params_path = datasets[0].flame_path_from_image_path(Path(singles[i]['image_path']))
        out_path = Path(config.train.run_dir)/ flame_track_params_path.relative_to(flame_track_params_path.parents[2])
        out_path.parent.mkdir(exist_ok=True, parents=True)
        flame_fits_ = dict([(k, v[i%v.shape[0]].cpu().numpy()) for k, v in flame_fits.items()])
        out_flame_fits = convert_flame_fits_to_dataset_flame_params(flame_fits_)
        np.savez(out_path, **out_flame_fits)


    # visualizing sequence of aligned gaussians:
    flame_vertices_unposed = unpose_vertices(flame_vertices_fit, flame_fits=flame_fits, flame=flame)
    out_path = config.train.run_dir + '/vis.mp4'
    K_static = singles[0]['K'].clone()
    H, W = singles[0]['image'].shape[-2:]
    K_static[..., 0, 2] = W/2
    K_static[..., 1, 2] = H/2
    Rt_static = singles[0]['cam_RT']
    Rt_static = torch.tensor([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]]).to(flame_vertices_unposed)

    for i in tqdm(list(range(len(gaussians['geometry']))), desc='Visualizing flame fits'):
        batch = none_collate([singles[i]])
        batch['root_RT'] = torch.eye(4).to(flame_vertices_fit)[None]
        unposed_batch = copy.deepcopy(batch)
        unposed_batch['K'] = K_static[None]
        unposed_batch['cam_RT'] = Rt_static[None]
        image = batch['image']
        B, C, H, W = image.shape
        renderer.resize(H, W)
        match_render = original_gaussians_renders[i][None]
        mesh_fit = Mesh(flame_vertices_fit[i:i+1], torch.from_numpy(flame.faces).long().to(flame_vertices_fit.device))
        mesh_unposed = Mesh(flame_vertices_unposed[i:i+1], torch.from_numpy(flame.faces).long().to(flame_vertices_fit.device))
        flame_fit_render = render_mesh(batch, mesh_fit)
        flame_unpose_render = render_mesh(unposed_batch, mesh_unposed)
        gaussians_ = AttrDict(dict([(k, v[i:i+1].cuda()) for k, v in gaussians.items()]))
        flame_fits_ = dict([(k, v[i:i+1] if k!= 'shape_params' else v) for k, v in flame_fits.items()])
        gaussians_unposed_ = AttrDict(unpose_gaussians(gaussians=gaussians_, flame_fits=flame_fits_, flame=flame, tex_to_mesh=tex_to_mesh))

        pred_image, render_pkg, pred_alpha, bg_color = splat(unposed_batch, gaussians_unposed_, bg_color='white', twoDgs=twoDgs, to_canonical=True)

        vis_img = torch.cat([image, match_render.to(image), flame_fit_render, pred_image, flame_unpose_render], dim=-1)[0]
        out_imgpath = f'{out_path}_frames/{i:06d}.jpg'
        Path(out_imgpath).parent.mkdir(exist_ok=True, parents=True)
        save_image(vis_img, out_imgpath)
        # save_image(mesh_render, f'/is/cluster/mprinzler/projects/gintern/GEM/experiments/gem/UNION_INQ807_TEETHFOCUS/debug/pca/aligned_gaussians/flame_{i:06d}.jpg')
        # save_image(mesh_render*.5+pred_image*.5, f'/is/cluster/mprinzler/projects/gintern/GEM/experiments/gem/UNION_INQ807_TEETHFOCUS/debug/pca/aligned_gaussians/overlay_{i:06d}.jpg')

    src = str(Path(out_imgpath).parent / '*.jpg')
    (
        ffmpeg
        .input(src, pattern_type='glob', framerate=10)
        .output(out_path)
        .overwrite_output()
        .run()
    )
    if Path(out_path).exists():
        shutil.rmtree(Path(out_imgpath).parent)

if __name__ == "__main__":
    path = sys.argv[1]
    config = OmegaConf.load(path)

    seed_everything()

    quantize = False
    debug_frames = None
    run(config, quantize, debug_frames=debug_frames)
