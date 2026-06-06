# fit_flame_pc_keypoint_warmup.py
# -----------------------------------------------------------------------------
# FLAME → point cloud fitting with 3 stages:
#   (1) Rt from 5 keypoints (PC_KEYPOINTS ↔ FLAME_KEYPOINTS)
#   (2) shape-only on the same 5 keypoints
#   (3) dense Chamfer (block-wise) on full point cloud; optimize the rest
# -----------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Dict, Sequence
import os, math
import numpy as np
import torch
import torch.nn as nn
import trimesh
from lib.F3DMM.FLAME2023.flame import FLAME
import torch.nn.functional as F
from lib.common import vis_3d_point_clouds
import pudb
# --- project imports (adjust paths to yours) ---
from lib.F3DMM.masks.masking import Masking


def save_xyz(filename, pc):
    pc = pc.cpu().numpy()
    np.savetxt(filename, pc, fmt='%.6f', delimiter=' ', header='x y z', comments='')


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_axis_angle(Rs):
    assert Rs.shape[-2:] == (3, 3)

    # Trace → angle
    trace = Rs[..., 0, 0] + Rs[..., 1, 1] + Rs[..., 2, 2]
    cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # Axis from skew-symmetric part
    axis = torch.stack([
        Rs[..., 2, 1] - Rs[..., 1, 2],
        Rs[..., 0, 2] - Rs[..., 2, 0],
        Rs[..., 1, 0] - Rs[..., 0, 1]
    ], dim=-1)

    denom = 2 * torch.sin(theta).unsqueeze(-1)
    axis = axis / torch.where(denom.abs() < 1e-8, torch.ones_like(denom), denom)

    axis = axis / axis.norm(dim=-1, keepdim=True)

    # Rotation vector = axis * angle
    rotvec = axis * theta.unsqueeze(-1)
    return rotvec


def to_rot(d6: torch.Tensor) -> torch.Tensor:
    mat = rotation_6d_to_matrix(d6)
    return matrix_to_axis_angle(mat)


def nn_pair_loss(A: torch.Tensor,
                 B: torch.Tensor,
                 squared: bool = True,
                 reduction: str = "mean"):
    """
    A: (N,3) source cloud
    B: (M,3) target cloud
    1) For each a_i in A, find its nearest neighbor b_j in B
    2) Compute distance on those pairs and reduce

    Returns:
      loss (scalar), idx (N,), B_matched (N,3)
    """
    # pairwise distances (1,N,M)
    D = torch.cdist(A.unsqueeze(0), B.unsqueeze(0), p=2)
    if squared:
        D = D.pow(2)

    # nearest neighbor in B for each point in A
    idx = D.argmin(dim=2).squeeze(0)            # (N,)
    B_matched = B[idx]                           # (N,3)

    # per-point distances on matched pairs
    if squared:
        per_point = ((A - B_matched) ** 2).sum(dim=1)  # squared L2
    else:
        per_point = (A - B_matched).norm(dim=1)        # L2

    if reduction == "sum":
        loss = per_point.sum()
    elif reduction == "none":
        loss = per_point
    else:  # "mean"
        loss = per_point.mean()

    return loss, idx, B_matched


# ------------------------------ CONFIG -------------------------------------- #

@dataclass
class FitConfig:
    # iters per stage
    iters_rt: int = 1000
    iters_dense: int = 1500

    # per-parameter learning rates
    lr_rot: float   = 0.15     # global_orient
    lr_trans: float = 0.025      # transl (depends on units)
    lr_shape: float = 0.005
    lr_expr: float  = 0.005
    lr_pose: float  = 0.1     # pose_rest, neck, eyes

    # Chamfer sampling + memory
    n_samples_flame: int = 10000
    x_block: int = 4096         # chamfer blocks
    y_block: int = 4096
    target_subsample: Optional[int] = 20000

    # Regularization
    w_shape: float = 1e-3
    w_expr: float  = 1e-4
    w_pose_rest: float = 5e-3
    w_neck_eye: float  = 1e-2

    # robust loss (keypoints)
    huber_delta: float = 0.01

    # logging
    log_every: int = 200


# --------------- DEFAULT 5-POINT CORRESPONDENCE (your lists) ---------------- #

PC_KEYPOINTS     = [20824, 20902, 29823, 39528, 39568, 36221, 44417, 47996]   # indexes in point cloud
FLAME_KEYPOINTS  = [2440,  1170,  3526,  2827,  1576, 3543, 3506, 3404]      # vertex IDs in FLAME


# -------------------------------- UTILS ------------------------------------- #

def to_torch(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def robust_huber(x, delta=0.01):
    ax = x.abs()
    return torch.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta)).mean()

def face_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    v0 = vertices[faces[:, 0]]; v1 = vertices[faces[:, 1]]; v2 = vertices[faces[:, 2]]
    return 0.5 * torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)

def sample_points_on_mesh(vertices: torch.Tensor, faces: torch.Tensor,
                          n_samples: int, exclude_vertex_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    device = vertices.device
    faces_use = faces
    if exclude_vertex_mask is not None:
        keep_face = (~exclude_vertex_mask[faces]).all(dim=1)
        faces_use = faces[keep_face]
        if faces_use.numel() == 0:
            return vertices.new_zeros((0, 3))
    areas = face_areas(vertices, faces_use) + 1e-12
    probs = areas / areas.sum()
    idx = torch.multinomial(probs, num_samples=n_samples, replacement=True)
    tri = faces_use[idx]
    v0 = vertices[tri[:, 0]]; v1 = vertices[tri[:, 1]]; v2 = vertices[tri[:, 2]]
    r1 = torch.rand(n_samples, device=device); r2 = torch.rand(n_samples, device=device)
    s1 = torch.sqrt(r1)
    w0 = 1 - s1; w1 = s1*(1 - r2); w2 = s1*r2
    return v0*w0[:,None] + v1*w1[:,None] + v2*w2[:,None]



def load_pointcloud_any(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".xyz":
        pts = []
        with open(path, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"): continue
                cols = s.split()
                if len(cols) >= 3:
                    try:
                        pts.append((float(cols[0]), float(cols[1]), float(cols[2])))
                    except: pass
        if not pts: raise ValueError(f"No points parsed from {path}")
        return np.asarray(pts, dtype=np.float32)
    obj = trimesh.load(path, process=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj.vertices.astype(np.float32)
    elif hasattr(obj, "vertices"):
        return np.asarray(obj.vertices, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported pc file: {path}")


# ---------------------------- FLAME WRAPPER --------------------------------- #

class FlameFitter(nn.Module):
    def __init__(self, nframes:int, n_shape=300, n_expr=100, device=None, static_eyes=False, static_eyelids=False):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.flame = FLAME().to(self.device).eval()

        self.global_orient = nn.Parameter(torch.zeros(nframes, 3, device=self.device))
        self.jaw_pose      = nn.Parameter(torch.zeros(nframes, 3, device=self.device))
        self.neck_pose     = nn.Parameter(torch.zeros(nframes, 3, device=self.device))
        self.transl        = nn.Parameter(torch.zeros(nframes, 3, device=self.device))

        self.shape = nn.Parameter(torch.zeros(1, n_shape, device=self.device))
        self.expr  = nn.Parameter(torch.zeros(nframes, n_expr, device=self.device))
        
        self.eye_pose =  torch.zeros(nframes, 6, device=self.device)
        # self.eyelids = torch.zeros(nframes, 2, device=self.device)

        if not static_eyes:
            self.eye_pose = nn.Parameter(self.eye_pose)
        # if not static_eyelids: 
        #     self.eyelids = nn.Parameter(self.eyelids)

        self.faces = self.flame.faces_tensor.to(self.device)

    def compose_pose_params(self):
        return torch.cat([self.global_orient, self.jaw_pose], dim=1)

    def forward(self):
        shape = self.shape.expand((self.expr.shape[0], self.shape.shape[1]))
        verts, _, _, _ = self.flame(
            shape_params=shape,
            expression_params=self.expr,
            pose_params=self.compose_pose_params(),
            neck_pose=self.neck_pose,
            eye_pose=self.eye_pose,
            transl=self.transl,
            delta=None,
        )
        return verts, self.faces

    def export_parameters(self) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            return {
                "shape_params": self.shape.detach().clone(),
                "expression_params": self.expr.detach().clone(),
                "pose_params": torch.cat([self.global_orient.detach().clone(), self.jaw_pose.detach().clone()], dim=-1),
                "neck_pose": self.neck_pose.detach().clone(),
                "eye_pose": self.eye_pose.detach().clone(),
                # "eyelids": self.eyelids.detach().clone(),
                "transl": self.transl.detach().clone(),
            }


# -------------------------- MAIN: 3-STAGE FITTER ---------------------------- #
def fit_flame_to_flame_vertices(
    target_vertices: torch.Tensor,                                # (B, V, 3)
    mask: Optional[torch.Tensor] = None,  # bool[V] True=ignore in FLAME sampling
    n_shape: int = 300,
    n_expr: int = 100,
    device: Optional[torch.device] = None,
    config: FitConfig = FitConfig(),
    static_eyes = False,
    static_eyelids = False,
) -> Dict[str, torch.Tensor]:

    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tP = target_vertices.to(dev)
    if mask is not None:
        mask = mask.to(dev)
    fitter = FlameFitter(n_shape=n_shape, n_expr=n_expr, device=dev, static_eyes=static_eyes, nframes=len(target_vertices))

    # ---------- Stage 1: Rt optim only ----------
    opt_rt = torch.optim.Adam(
        [
            {"params": [fitter.global_orient], "lr": config.lr_rot},
            {"params": [fitter.transl],        "lr": config.lr_trans},
        ]
    )
    for it in range(config.iters_rt):
        opt_rt.zero_grad()
        Vpred, _ = fitter()
        Vtgt = tP
        if mask is not None:
            Vpred = Vpred[:, mask]
            Vtgt = Vtgt[:, mask]
        

        L = robust_huber(Vpred - Vtgt, delta=config.huber_delta)
        L.backward()
        opt_rt.step()
        if (it + 1) % config.log_every == 0:
            print(f"[Stage1 Rt {it+1:04d}/{config.iters_rt}] L={L.item():.6f}")
            # # vis pred vs gt
            # vis_idx = 22
            # vis_3d_point_clouds(dict(pred=Vpred[vis_idx].detach().cpu().numpy(), gt=Vtgt[vis_idx].detach().cpu().numpy()), f'demos/optim_flame_rt_{it:06d}.html')


    # ---------- Stage 2: DENSE (opt. all the rest) ----------
    param_groups = [
            {"params": [fitter.transl],                         "lr": config.lr_trans*0.1},
            {"params": [fitter.global_orient],                  "lr": config.lr_rot*0.1},
            {"params": [fitter.neck_pose],                      "lr": config.lr_pose*0.1},
            {"params": [fitter.jaw_pose],                      "lr": config.lr_pose*0.1},
            {"params": [fitter.shape],                      "lr": config.lr_shape*0.1},
            {"params": [fitter.expr],                      "lr": config.lr_expr*0.1},
        ]
    if not static_eyes:
        param_groups.append({"params": [fitter.eye_pose],                      
                             "lr": config.lr_pose * 0.1})
    # if not static_eyelids:
    #     param_groups.append({"params": [fitter.eyelids],                      
    #                          "lr": config.lr_pose * 0.1})
        
    opt_dense = torch.optim.Adam(param_groups)

    for it in range(config.iters_dense):
        opt_dense.zero_grad()
        Vpred, _ = fitter()
        Vtgt = tP
        if mask is not None:
            Vpred = Vpred[:, mask]
            Vtgt = Vtgt[:, mask]
        L = robust_huber(Vpred - Vtgt, delta=config.huber_delta)
        loss = L
        loss.backward()
        # optional: clip rotations
        torch.nn.utils.clip_grad_norm_([fitter.global_orient, fitter.jaw_pose, fitter.neck_pose, fitter.eye_pose], max_norm=1.0)
        opt_dense.step()

        if (it + 1) % config.log_every == 0:
            print(f"[Stage4 Dense {it+1:04d}/{config.iters_dense}] "
                  f"L={L.item():.6f}")
            # vis_idx = 22
            # vis_3d_point_clouds(dict(pred=Vpred[vis_idx].detach().cpu().numpy(), gt=Vtgt[vis_idx].detach().cpu().numpy()), f'demos/optim_flame_dense_{it:06d}.html')


    # # vis fitting temporal stability
    # with torch.no_grad():
    #     Vpred, _ = fitter()
    #     Vtgt = tP
    #     if mask is not None:
    #         Vpred = Vpred[:, mask]
    #         Vtgt = Vtgt[:, mask]

    #     Vpred_x = Vpred[..., 0]
    #     Vpred_y = Vpred[..., 1]
    #     Vtgt_x = Vtgt[..., 0]
    #     Vtgt_y = Vtgt[..., 1]

    #     xmax = torch.max(torch.cat((Vpred_x, Vtgt_x), dim=0).flatten()).cpu().item()
    #     xmin = torch.min(torch.cat((Vpred_x, Vtgt_x), dim=0).flatten()).cpu().item()
    #     ymax = torch.max(torch.cat((Vpred_y, Vtgt_y), dim=0).flatten()).cpu().item()
    #     ymin = torch.min(torch.cat((Vpred_y, Vtgt_y), dim=0).flatten()).cpu().item()

    #     import matplotlib.pyplot as plt
    #     for i in range(len(Vpred_x)):
    #         fig = plt.figure(figsize=(15, 15))
    #         plt.scatter(Vpred_x[i].cpu().numpy(), Vpred_y[i].cpu().numpy(), s=.5)
    #         plt.scatter(Vtgt_x[i].cpu().numpy(), Vtgt_y[i].cpu().numpy(), s=.5)
    #         plt.xlim(xmin, xmax)
    #         plt.ylim(ymin, ymax)
    #         plt.savefig(f'demos/flame_fitting_vis_{i:06d}.jpg')
    #         plt.close(fig)
    #     pudb.set_trace()



    return fitter.export_parameters()


def fit_flame_to_pointcloud(
    target_points: np.ndarray,                                # (N,3)
    *,
    pc_keypoints_idx: Sequence[int] = PC_KEYPOINTS,           # indexes into target_points
    flame_keypoints_idx: Sequence[int] = FLAME_KEYPOINTS,     # FLAME vertex IDs
    flame_sampling_ignore_mask: Optional[np.ndarray] = None,  # bool[V] True=ignore in FLAME sampling
    target_point_ignore_mask: Optional[np.ndarray] = None,    # bool[N] True=ignore target points
    n_shape: int = 300,
    n_expr: int = 100,
    device: Optional[torch.device] = None,
    config: FitConfig = FitConfig(),
    tracking = None
) -> Dict[str, torch.Tensor]:
    """WARNING: UNTESTED!"""

    dev = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tP_all = to_torch(target_points, dev)

    # Optional target ignore
    if target_point_ignore_mask is not None:
        keep = ~torch.as_tensor(target_point_ignore_mask, dtype=torch.bool, device=dev)
        if keep.sum() == 0:
            raise ValueError("All target points are ignored.")
        tP = tP_all[keep]
    else:
        tP = tP_all

    # Subsample for speed
    if config.target_subsample is not None and tP.shape[0] > config.target_subsample:
        idx = torch.randperm(tP.shape[0], device=dev)[: config.target_subsample]
        tP = tP[idx]

    # 5 keypoints from the ORIGINAL array (indices refer to original pc)
    pc_idx = torch.as_tensor(pc_keypoints_idx, dtype=torch.long, device=dev)
    if pc_idx.max().item() >= tP_all.shape[0]:
        raise ValueError("pc_keypoints_idx contains an index >= number of target points.")
    KP_target = tP_all[pc_idx]  # (5,3)

    fitter = FlameFitter(n_shape=n_shape, n_expr=n_expr, device=dev)

    with torch.no_grad():
        fitter.global_orient.copy_(torch.tensor([[math.pi, 0.0, 0.0]], device=dev))
        V0, _ = fitter()                      # vertices with the flipped rotation
        c_flame = V0.mean(dim=0)              # centroid of rotated FLAME
        c_target = tP.mean(dim=0)
        fitter.transl.copy_((c_target - c_flame).unsqueeze(0))  # translation compensation

        if tracking is not None:
            fitter.expr.copy_(tracking["exp"][0])
            fitter.shape.copy_(tracking["shape"][0])
            # fitter.eyelids.copy_(tracking["eyelids"][0])

            left = to_rot(tracking["eyes"][0][:, 0:6])
            right = to_rot(tracking["eyes"][0][:, 6:12])
            eyes = torch.cat((left, right), dim=1)

            fitter.eye_pose.copy_(eyes)
            fitter.jaw_pose.copy_(to_rot(tracking["jaw"][0]))

    flame_ignore = None
    if flame_sampling_ignore_mask is not None:
        flame_ignore = torch.as_tensor(flame_sampling_ignore_mask, dtype=torch.bool, device=dev)

    # ---------- Stage 1: Rt from 5 keypoints ----------
    opt_rt = torch.optim.Adam(
        [
            {"params": [fitter.global_orient], "lr": config.lr_rot},
            {"params": [fitter.transl],        "lr": config.lr_trans},
        ]
    )
    kp_vidx = torch.as_tensor(flame_keypoints_idx, dtype=torch.long, device=dev)
    for it in range(config.iters_rt):
        opt_rt.zero_grad()
        Vpred, _ = fitter()
        KP_pred = Vpred[kp_vidx]                        # (5,3)
        L_kp = robust_huber(KP_pred - KP_target, delta=config.huber_delta)
        L_kp.backward()
        opt_rt.step()
        if (it + 1) % config.log_every == 0:
            print(f"[Stage1 Rt {it+1:04d}/{config.iters_rt}] kp={L_kp.item():.6f}")


    # ---------- Stage 2: DENSE Chamfer (opt. all the rest) ----------
    opt_dense = torch.optim.Adam(
        [
            {"params": [fitter.transl],                         "lr": config.lr_trans * 0.1},
            {"params": [fitter.global_orient],                  "lr": config.lr_rot * 0.1},
            {"params": [fitter.neck_pose],                      "lr": config.lr_rot * 0.1},
        ]
    )

    for it in range(config.iters_dense):
        opt_dense.zero_grad()

        Vpred, Fpred = fitter()
        flame_pc = sample_points_on_mesh(
            vertices=Vpred, faces=Fpred, n_samples=config.n_samples_flame,
            exclude_vertex_mask=flame_ignore
        )
        if flame_pc.numel() == 0:
            raise ValueError("FLAME sampling produced 0 points (check mask).")

        KP_pred = Vpred[kp_vidx]
        L_ch = nn_pair_loss(flame_pc, tP)[0]

        loss = L_ch
        loss.backward()
        # optional: clip rotations
        torch.nn.utils.clip_grad_norm_([fitter.global_orient, fitter.jaw_pose, fitter.neck_pose, fitter.eye_pose], max_norm=1.0)
        opt_dense.step()

        if (it + 1) % config.log_every == 0:
            print(f"[Stage4 Dense {it+1:04d}/{config.iters_dense}] "
                  f"ch={L_ch.item():.6f} | kp={L_kp.item():.6f}")


    return fitter.export_parameters()


# --------------------------------- Example ---------------------------------- #
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) load a point cloud (.xyz/.ply/.obj)
    pc_path = "sandbox/gaussians.pt"
    # target_points = load_pointcloud_any(pc_path)
    gaussians = torch.load("sandbox/gaussians.pt")
    mask = gaussians["mask"]
    mask = mask.squeeze(0).squeeze(0).reshape(-1).bool()

    # def extract(x: torch.Tensor) -> torch.Tensor:
    #     x_hw_c = x.squeeze(0).reshape(x.shape[1], -1).transpose(0, 1)  # (HW, C)
    #     return x_hw_c[mask]

    # target_points = extract(gaussians["xyz"])
    # target_points = (target_points - target_points.mean(0)) @ torch.diag(torch.tensor([-1.,1.,1.], device=target_points.device)) + target_points.mean(0)


    ava_template = load_obj('/home/mprinzler/projects/gintern/gtempeh/assets/ava256/face_topology_cleaned.obj')
    ava_verts = ava_template['v']
    ava_verts = torch.from_numpy(ava_verts).float()
    target_points = ava_verts


    
    # tracking = torch.load("sandbox/tracking.frame", weights_only=False, map_location="cpu")["flame"]
    # tracking = {k: torch.from_numpy(v)[None] for k, v in tracking.items()}
    tracking=None

    masking = Masking()

    mask = ~masking.face()

    # 3) fit
    params = fit_flame_to_pointcloud(
        target_points=target_points,
        n_shape=300, n_expr=100,
        device=device,
        tracking=tracking,
        flame_sampling_ignore_mask=mask.cpu().numpy()
    )

    # 4) reconstruct and export the fitted FLAME mesh
    flame = FLAME().to(device).eval()
    with torch.no_grad():
        V_fit, _, _, _ = flame(
            shape_params=params["shape_params"],
            expression_params=params["expression_params"],
            pose_params=params["pose_params"],
            neck_pose=params["neck_pose"],
            eye_pose=params["eye_pose"],
            transl=params["transl"],
            # eyelids=params["eyelids"],
        )
    V_fit = V_fit[0].cpu().numpy()
    F_fit = flame.faces_tensor.cpu().numpy()
    trimesh.Trimesh(vertices=V_fit, faces=F_fit, process=False).export("sandbox/fitted_from_pc.obj")
    save_xyz("sandbox/pc.xyz", target_points)
