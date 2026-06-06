import matplotlib.pyplot as plt
import torch

all_extrinsics = torch.load('/tmp/camera_extrinsics.pt')[0]  # N x 3 x 4
all_verts = torch.load('/tmp/v_reg_global.pt')[0]
ray_o =torch.load('/tmp/ray_o.pt')[0]
ray_d =torch.load('/tmp/ray_d.pt')[0]

rots_cam2world = all_extrinsics[:, :3, :3].permute(0, 2, 1)  # N, 3, 3
cam_centers = (-rots_cam2world @ all_extrinsics[:, :3, -1:])  # N, 3, 1
cam_ids = list(range(len(all_extrinsics)))

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
s = 50

# Drawing Cameras
for i, color in enumerate(["red", "green", "blue"]):
    ax.quiver(cam_centers[:, 0, 0],
              cam_centers[:, 1, 0],
              cam_centers[:, 2, 0],
              s * rots_cam2world[:, 0, i],
              s * rots_cam2world[:, 1, i],
              s * rots_cam2world[:, 2, i],
              edgecolor=color)

# Annotating Cameras
for i, id in enumerate(cam_ids):
    ax.text(cam_centers[i, 0, 0],
            cam_centers[i, 1, 0],
            cam_centers[i, 2, 0],
            str(id))

# draw object
ax.scatter(all_verts[:, 0], all_verts[:, 1], all_verts[:, 2])

# draw rays
s_ray = 1000
import numpy as np
idcs = np.random.permutation(ray_o.shape[-1] * ray_o.shape[-2])[:1000]
ax.quiver(ray_o[0, 0].flatten()[idcs], ray_o[0, 1].flatten()[idcs], ray_o[0, 2].flatten()[idcs],
          s_ray*ray_d[0, 0].flatten()[idcs], s_ray*ray_d[0, 1].flatten()[idcs], s_ray*ray_d[0, 2].flatten()[idcs])
idcs = np.arange(100)
ax.quiver(ray_o[0, 0].flatten()[idcs], ray_o[0, 1].flatten()[idcs], ray_o[0, 2].flatten()[idcs],
          s_ray*ray_d[0, 0].flatten()[idcs], s_ray*ray_d[0, 1].flatten()[idcs], s_ray*ray_d[0, 2].flatten()[idcs], edgecolor="C1")


# setting correct and equally scaled view frustum
bbx = cam_centers[..., 0].min(dim=0).values, cam_centers[..., 0].max(dim=0).values
bbx_side_length = torch.max(bbx[1] - bbx[0])
bbx_center = 0.5 * (bbx[0] + bbx[1])
co_min = bbx_center - bbx_side_length / 2
co_max = bbx_center + bbx_side_length / 2
ax.set_xlim(co_min[0], co_max[0])
ax.set_ylim(co_min[1], co_max[1])
ax.set_zlim(co_min[2], co_max[2])

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()