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


from collections import namedtuple
import torch as th
import torch.nn.functional as F
import trimesh
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_to_matrix, quaternion_multiply
import pudb
import torch
import copy
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import einops
from utils.geometry import AttrDict
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, quaternion_raw_multiply, standardize_quaternion
from typing import Any, Union
Mesh = namedtuple("Mesh", "v f")


def batchify_flame_params(flame_params:dict[str, Any]):
    batchified_flame_params = dict()
    for k, v in flame_params.items():
        if len(v.shape) == 1 or (k == 'static_offset' and len(v.shape)==2):
            v = v[None]
        batchified_flame_params[k] = v
    return batchified_flame_params

def convert_flame_fits_to_dataset_flame_params(flame_fits, inverse=False):
    is_batch = len(next(iter(flame_fits.values())).shape) == 2
    if not is_batch:
        flame_fits = dict([(k, v[None]) for k, v in flame_fits.items()])
    v = next(iter(flame_fits.values()))
    B = v.shape[0]
    is_tensor = isinstance(v, torch.Tensor)
    device = v.device if is_tensor else None
    
    if inverse:
        ret = dict(
                    transl = flame_fits['translation'],
                    pose_params = torch.cat([flame_fits['rotation'], flame_fits['jaw_pose']], dim=-1) if is_tensor else np.concatenate([flame_fits['rotation'], flame_fits['jaw_pose']], axis=-1),
                    neck_pose = flame_fits['neck_pose'],
                    eye_pose = flame_fits['eyes_pose'],
                    shape_params = flame_fits['shape'],
                    expression_params = flame_fits['expr'],
                    static_offset=flame_fits.get('static_offset' ,torch.zeros((B, 5023, 3), dtype=torch.float32, device=device) if is_tensor else np.zeros((B, 5023, 3), dtype=np.float32)),
                )
    else:
        ret = dict(
                    translation = flame_fits['transl'],
                    rotation=flame_fits['pose_params'][:, :3],
                    neck_pose=flame_fits['neck_pose'],
                    jaw_pose=flame_fits['pose_params'][:, 3:],
                    eyes_pose=flame_fits['eye_pose'],
                    shape=flame_fits['shape_params'],
                    expr=flame_fits['expression_params'],
                    static_offset=flame_fits.get('static_offset' ,torch.zeros((B, 5023, 3), dtype=torch.float32, device=device) if is_tensor else np.zeros((B, 5023, 3), dtype=np.float32)),
                )
    if not is_batch:
        ret = dict([(k, v[0]) for k, v in ret.items()])
    return ret

def unpose_gaussians(gaussians, flame_fits,  flame, tex_to_mesh):

    # TODO remove neck joint (non-trivial because of invertibility issues with LBS)
    # A_with_transl_headcentered = einops.einsum(inverse_root_RT, A_with_transl, 'b i j, b n j k -> b n j k')
    # gaussians['geometry'] = remove_joint(gaussians['geometry'].to(A), A_rel_to_root, W, tex_to_mesh, joint=1).cpu()  # TODO

    flame_fits = dict([(k, v) for k, v in flame_fits.items() if k != 'static_offset'])
    flame_fits['eye_pose'] = flame_fits['eye_pose'] * 0  # not using eye pose
    flame_vertices, J, A, W = flame(**flame_fits)
    A_with_transl = A.clone()
    A_with_transl[:, :, :3, -1] += flame_fits['transl'].unsqueeze(1)

    face_root_RT = A_with_transl[:, 1]
    face_root_RT_inv = invert_c2w(face_root_RT)
    ref_quaternion = standardize_quaternion(matrix_to_quaternion(face_root_RT_inv[0, :3,:3]))
    T = einops.einsum(W, A_with_transl, 'b n j, b j c1 c2 -> b n c1 c2')  # (B, nverts, 4, 4)
    T_inv = torch.linalg.inv(T)
    T_inv_gauss = T_inv[:, tex_to_mesh]  # (B, ngauss, 4, 4)
    
    # every gaussian has an individual trafo so treat gaussian splat sets as batches of gaussian sets with 1 gaussian each
    B, G = T_inv_gauss.shape[:2]
    gaussians = dict([(k, einops.rearrange(v, 'b g c -> (b g) 1 c') if k != 'tex_map' else v) for k, v in gaussians.items()])
    T_inv_gauss = einops.rearrange(T_inv_gauss, 'b g c1 c2 -> (b g) c1 c2')
    gaussians = rigid_trafo_gaussians(gaussians, T_inv_gauss, ref_quaternion=ref_quaternion)  # to canonical space

    # join gaussian sets again
    gaussians = dict([(k, einops.rearrange(v, '(b g) 1 c -> b g c', b=B, g=G) if k != 'tex_map' else v) for k, v in gaussians.items()])
    return gaussians 


def unpose_vertices(xyz, flame_fits,  flame):
    '''
    Args:
        xyz: (B, N, 3)
    '''

    flame_fits = dict([(k, v) for k, v in flame_fits.items() if k != 'static_offset'])
    flame_fits['eye_pose'] = flame_fits['eye_pose'] * 0  # not using eye pose
    flame_vertices, J, A, W = flame(**flame_fits)
    A_with_transl = A.clone()
    A_with_transl[:, :, :3, -1] += flame_fits['transl'].unsqueeze(1)

    face_root_RT = A_with_transl[:, 1]
    face_root_RT_inv = invert_c2w(face_root_RT)
    ref_quaternion = standardize_quaternion(matrix_to_quaternion(face_root_RT_inv[0, :3,:3]))
    T = einops.einsum(W, A_with_transl, 'b n j, b j c1 c2 -> b n c1 c2')  # (B, nverts, 4, 4)
    T_inv = torch.linalg.inv(T)

    xyz_hom = torch.cat((xyz, torch.ones_like(xyz[..., :1])), dim=-1)
    xyz_unposed = einops.einsum(T_inv, xyz_hom, 'b n i j, b n j -> b n i')[..., :3]
    return xyz_unposed


def pose_gaussians(gaussians, flame_fits, flame, tex_to_mesh):
    flame_fits = dict([(k, v) for k, v in flame_fits.items() if k != 'static_offset'])
    flame_fits['eye_pose'] = flame_fits['eye_pose'] * 0  # not using eye pose
    flame_vertices, J, A, W = flame(**flame_fits)
    A_with_transl = A.clone()
    A_with_transl[:, :, :3, -1] += flame_fits['transl'].unsqueeze(1)

    face_root_RT = A_with_transl[:, 1]
    ref_quaternion = standardize_quaternion(matrix_to_quaternion(face_root_RT[0, :3,:3]))
    T = einops.einsum(W, A_with_transl, 'b n j, b j c1 c2 -> b n c1 c2')  # (B, nverts, 4, 4)
    T_gauss = T[:, tex_to_mesh]  # (B, ngauss, 4, 4)
    
    # every gaussian has an individual trafo so treat gaussian splat sets as batches of gaussian sets with 1 gaussian each
    B, G = T_gauss.shape[:2]
    gaussians = dict([(k, einops.rearrange(v, 'b g c -> (b g) 1 c') if k != 'tex_map' else v) for k, v in gaussians.items()])
    T_gauss = einops.rearrange(T_gauss, 'b g c1 c2 -> (b g) c1 c2')
    gaussians = rigid_trafo_gaussians(gaussians, T_gauss, ref_quaternion=ref_quaternion)  # to canonical space

    # join gaussian sets again
    gaussians = dict([(k, einops.rearrange(v, '(b g) 1 c -> b g c', b=B, g=G)  if k != 'tex_map' else v) for k, v in gaussians.items()])
    return gaussians 



def slice_dict(d, idx):
    ret_dict = {}
    for k, v in d:
        try:
            v = v[idx]
        except (KeyError):
            v = slice_dict(v, idx)
        ret_dict[k] = v
    return ret_dict

def dict_to_numpy(d):
    ret_dict = {}
    for k, v in d.items():
        if isinstance(v, th.Tensor):
            v = v.detach().cpu().numpy()
        elif isinstance(v, dict) or isinstance(v, AttrDict):
            v = dict_to_numpy(v)
        else:
            raise NotImplementedError
        ret_dict[k] = v
    return ret_dict


def invert_c2w(c2w: Union[torch.Tensor,np.ndarray]) -> torch.Tensor:
  """Inverts a camera to world transform.

  Args:
    c2w: The camera to world transform. (..., 4, 4)

  Returns:
    The inverted camera to world transform. (..., 4, 4)
  """
  w2c = torch.zeros_like(c2w) if isinstance(c2w, torch.Tensor) else np.zeros_like(c2w)
  w2c[..., 3, 3] = 1
  w2c[..., :3, :3] = c2w[..., :3, :3].transpose(-1, -2)
  w2c[..., :3, -1:] = -c2w[..., :3, :3].transpose(-1, -2) @ c2w[..., :3, -1:]
  return w2c

def vis_3d_point_clouds(named_point_clouds: dict[str, np.ndarray], outpath: str):
    """
    Visualizes multiple 3D point clouds with different colors in a single 3D plot, and saves as HTML.

    Args:
        named_point_clouds: Dictionary mapping names to point clouds as np.ndarray of shape (N,3)
        outpath: Path to save the HTML visualization
    """
    fig = go.Figure()

    # Get a color palette large enough for all point clouds
    colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    
    for idx, (name, points) in enumerate(named_point_clouds.items()):
        color = colors[idx % num_colors]
        trace = go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(size=3, color=color),
            name=name
        )
        fig.add_trace(trace)

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        title='3D Point Clouds',
        legend=dict(itemsizing='constant')
    )
    
    fig.write_html(outpath)
    print(f"Visualization saved to {outpath}")

def make_vis_gaussians(gaussians):
    color_key = [k for k in ['colors', 'apperance'] if k in gaussians][0]
    vis_gaussians = copy.deepcopy(gaussians)
    rg = torch.Generator().manual_seed(0)
    colors = vis_gaussians[color_key]
    colors = torch.rand(size=colors.shape, generator=rg).to(colors)
    vis_gaussians[color_key] = colors
    return vis_gaussians


def parse_payload(results, root_RT):
    is_batch = len(results.geometry.shape) == 3
    if not is_batch:
        results = AttrDict(dict([(k, v.unsqueeze(0)) for k, v in results.items()]))
    means3D = results.geometry
    opacity = results.opacity
    scales = results.scales
    rotation = F.normalize(results.rotation, dim=-1)

    if root_RT is not None:
        R = root_RT[:, :3, :3]
        T = root_RT[:, :3, 3]
        means3D = einops.einsum(R, means3D, 'b i j, b n j -> b n i') + T.unsqueeze(1)

        N = means3D.shape[1]
        Q = einops.repeat(matrix_to_quaternion(R), 'b c -> b n c', n=N)
        rotation = quaternion_multiply(Q, rotation)

    pkg = AttrDict({
        "means3D": means3D,
        "scales": scales,
        "rotation": rotation,
        "opacity": opacity,
    })

    if "shs" in results:
        if is_batch:
            raise NotImplementedError('Not implemented for is_batch==True')
        pkg["shs"] = th.cat([results.colors[:, None, :], results.shs.reshape(-1, 15, 3)], dim=1)
        pkg["sh_degree"] = 3
    elif "shadow" in results:
        pkg["colors_precomp"] = results.apperance * (1.0 - th.clamp(results.shadow, 0, 1))
    elif 'apperance' in results:
        pkg["colors_precomp"] = results.apperance
    else:
        pkg["colors_precomp"] = results.colors

    if not is_batch:
        pkg = AttrDict(dict([(k, v.squeeze(0)) for k, v in pkg.items()]))

    return pkg

def markley_average_quaternions(quats: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Markley average of quaternions across batches.

    Args:
        quats: Tensor of shape (B, N, 4), assumed normalized quaternions.
        eps: Small epsilon to avoid numerical issues.

    Returns:
        mean_quats: Tensor of shape (N, 4), normalized mean quaternions.
    """
    assert quats.ndim == 3 and quats.shape[-1] == 4, "Expected (B, N, 4)"
    B, N, _ = quats.shape

    # Normalize to be safe
    quats = quats / (quats.norm(dim=-1, keepdim=True).clamp_min(eps))

    # Align quaternion signs per N across batches to avoid hemisphere flips
    quats = align_quaternions(quats, quats[:1].expand_as(quats))

    # Compute Markley average per quaternion group N
    # For each N, we build the 4x4 covariance matrix across B
    Q = quats.transpose(0, 1)  # (N, B, 4)
    M = torch.einsum("nbi,nbj->nij", Q, Q) / B  # (N, 4, 4)

    # Compute principal eigenvector (largest eigenvalue) of M
    eigvals, eigvecs = torch.linalg.eigh(M)  # (N, 4), (N, 4, 4)
    mean_quats = eigvecs[..., -1]  # (N, 4): eigenvector with largest eigenvalue

    # Normalize to ensure unit quaternions
    mean_quats = mean_quats / (mean_quats.norm(dim=-1, keepdim=True).clamp_min(eps))
    mean_quats = mean_quats.to(quats)

    return mean_quats


def average_gaussians(gaussians: dict):
    mean_gaussians = dict()
    for k, v in gaussians.items():
        if k == 'rotation':
            v = markley_average_quaternions(v)
        else:
            v = torch.mean(v, dim=0)
        mean_gaussians[k] = v
    return mean_gaussians


def align_quaternions(q: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Flip quaternions so they are consistently oriented with a reference quaternions.
    Args:
        q: (...,4) tensor of quaternions
        ref: (...,4) reference quaternions
    """
    dot = (q * ref).sum(-1, keepdim=True)  # alignment score
    flip_mask = (dot < 0).float()
    return q * (1 - 2 * flip_mask)



def rigid_trafo_gaussians(gaussians, RT, ref_quaternion=None):
    gaussians = copy.copy(gaussians)
    xyz = gaussians['geometry']
    rotation = gaussians['rotation']
    is_batch = len(xyz.shape) == 3
    if not is_batch:  # make batch first and then reduce dimension again
        xyz = xyz[None]
        RT = RT[None]
        rotation = rotation[None]
    RT = RT.to(xyz)
    xyz = einops.einsum(RT, 
                        torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1),
                        'b i j, b n j -> b n i')[..., :3]
    
    R_quat = matrix_to_quaternion(RT[:, :3,:3])
    if ref_quaternion is not None:
        ref_quaternion = ref_quaternion.to(xyz)
        R_quat = align_quaternions(R_quat, ref_quaternion)
    rotation = quaternion_raw_multiply(R_quat[:, None], rotation)

    if not is_batch:
        xyz = xyz[0]
        rotation = rotation[0]

    gaussians['geometry'] = xyz
    gaussians['rotation'] = rotation
    return gaussians


def remove_joint(xyz, A, W, tex_to_mesh, joint=0):
    is_batch = len(xyz.shape) == 3
    if not is_batch:
        xyz = xyz[None]
    B = len(xyz)

    A = A.clone()
    W = W.clone()
    A[:, joint, ...] = invert_c2w(A[:, joint, ...])

    for i in range(5):
        if i == joint:
            W[:, :, i] = 1
        else:
            W[:, :, i] = 0

    T = th.matmul(W, A.view(B, 5, 16)).view(B, -1, 4, 4)
    T = T[:, tex_to_mesh, ...]

    homogen_coord = th.ones([B, xyz.shape[1], 1]).cuda()
    v_posed_homo = th.cat([xyz, homogen_coord], dim=2)
    v_homo = th.matmul(T, th.unsqueeze(v_posed_homo, dim=-1))

    xyz = v_homo[:, :, :3, 0]

    if not is_batch:
        xyz = xyz[0]
    return xyz

def add_joint(xyz, A, W, tex_to_mesh, joint=0):
    is_batch = len(xyz.shape) == 3
    if not is_batch:
        xyz = xyz.unsqueeze(0)
        A = A.unsqueeze(0)
        W = W.unsqueeze(0)
    B = len(xyz)
    W = W.clone()
    A = A.squeeze(1)  # for some reason the individual samples of A have shape (1,5,4,4) so batches have shape (B, 1, 5, 4, 4)
    W = W.squeeze(1)
    W[:, :, :5] = 0
    W[:, :, joint] = 1

    T = th.matmul(W, A.view(B, 5, 16)).view(B, -1, 4, 4)
    T = T[:, tex_to_mesh, ...]

    homogen_coord = th.ones([B, xyz.shape[1], 1]).cuda()
    v_posed_homo = th.cat([xyz, homogen_coord], dim=2)
    v_homo = th.matmul(T, th.unsqueeze(v_posed_homo, dim=-1))

    xyz = v_homo[:, :, :3, 0]

    if not is_batch:
        xyz = xyz[0]
    return xyz

def to_trimesh(mesh):
    v = mesh.v.cpu().numpy()
    f = mesh.f.cpu().numpy()

    if len(v) == 3:
        v = v[0]
    if len(f) == 3:
        f = f[0]

    return trimesh.Trimesh(v, f, process=False)


def interpolate(t, size):
    return F.interpolate(t[None], size, mode="bilinear")[0]


def flatten(xyz):
    B, C, H, W = xyz.shape
    # return xyz.permute(0, 2, 3, 1).reshape(-1, C).contiguous().clone()

    w, h = th.meshgrid(
        th.arange(H, device="cuda"),
        th.arange(W, device="cuda"),
        indexing="xy",
    )

    idsH = th.flatten(h)
    idsW = th.flatten(w)

    extracted = xyz[:, :, idsH, idsW][0].permute(1, 0)

    return extracted
