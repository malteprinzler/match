from datasets.data_utils import visualize_sample_wojciech, visualize_sample, visualize_camera_grid, tempeh_unnormalize_image
from PIL import Image
import pudb
import numpy as np
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import einops
from gtempeh_utils import geo_util
import numpy as np
import plotly.graph_objects as go



def vis_scene_geometry(sample, cone_size=100.0):
    extrinsics = sample['stereo_camera_extrinsics']  # (V, 3, 4)
    camera_centers = sample['stereo_camera_centers']  # (V, 3)
    cam_poses = geo_util.invert_c2w(extrinsics)

    v_scan = sample['v_scan']  
    v_registration, f_registration = sample['v_registration'], sample['f_registration']
    v_reg_sampled, f_reg_sampled = sample['v_reg_sampled'], sample['f_reg_sampled']
    v_reg_global, f_reg_global = sample['v_reg_global'], sample['f_reg_global']

    fig = go.Figure()

    # --- Camera centers ---
    fig.add_trace(go.Scatter3d(
        x=camera_centers[:, 0],
        y=camera_centers[:, 1],
        z=camera_centers[:, 2],
        mode="markers",
        marker=dict(size=50, color="red"),
        name="Camera centers"
    ))

    # --- Camera coordinate axes (cones) ---
    axis_colors = ["red", "green", "blue"]
    for i, pose in enumerate(cam_poses):
        R = pose[:3, :3]  # rotation
        c = pose[:3, -1]
        
        for axis, color in zip(range(3), axis_colors):
            direction = R[:, axis] * cone_size
            fig.add_trace(go.Cone(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                u=[direction[0]],
                v=[direction[1]],
                w=[direction[2]],
                sizemode="absolute",
                anchor='tail',
                sizeref=100.0,
                showscale=False,
                colorscale=[[0, color], [1, color]],
                name=f"Cam {i} axis {axis}"
            ))

    # --- Meshes ---
    def add_mesh(vertices, faces, color, name, opacity=0.5):
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=color,
            opacity=opacity,
            name=name
        ))

    add_mesh(v_registration, f_registration, "orange", "Registration mesh")
    add_mesh(v_reg_sampled, f_reg_sampled, "purple", "Reg sampled mesh")
    add_mesh(v_reg_global, f_reg_global, "cyan", "Reg global mesh")

    # --- Scan points ---
    fig.add_trace(go.Scatter3d(
        x=v_scan[:, 0], y=v_scan[:, 1], z=v_scan[:, 2],
        mode="markers",
        marker=dict(size=1, color="gray"),
        name="Scan points"
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X"),
            yaxis=dict(title="Y"),
            zaxis=dict(title="Z")
        ),
        title="Scene Geometry Visualization"
    )
    return fig

def project_points(points, intrinsics, extrinsics):
    """ Project 3D points to 2D image coordinates. """
    points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)  # (N, 4)
    cam_points = (extrinsics @ points_h.T).T  # (N, 4)
    cam_points = cam_points[:, :3]
    proj = (intrinsics @ cam_points.T).T      # (N, 3)
    proj = proj[:, :2] / proj[:, 2:3]
    return proj

def vis_projections(sample):
    images = [tempeh_unnormalize_image(img) for img in einops.rearrange(sample['stereo_images'].numpy(), 'v c h w -> v h w c')]                     # (V, H, W, 3)
    images_aug = [tempeh_unnormalize_image(img) for img in einops.rearrange(sample['stereo_images_augmented'].numpy(), 'v c h w -> v h w c')]                     # (V, H, W, 3)
    intrinsics = sample['stereo_camera_intrinsics'].numpy()      # (V, 3, 3)
    intrinsics_aug = sample['stereo_camera_intrinsics_augmented'].numpy()  # (V, 3, 3)
    extrinsics = sample['stereo_camera_extrinsics'].numpy()      # (V, 4, 4)

    meshes = {
        "v_scan": sample['v_scan'].numpy(),
        "v_registration": sample['v_registration'].numpy(),
        "v_reg_sampled": sample['v_reg_sampled'].numpy(),
        "v_reg_global": sample['v_reg_global'].numpy(),
    }

    V = len(images)
    num_rows = 1 + len(meshes)   # 1 row for base image, rest for meshes
    num_cols = V * 2             # two columns per view (orig + aug)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6*num_cols, 3*num_rows))

    if num_rows == 1:
        axes = axes[None, :]  # keep 2D shape
    if num_cols == 1:
        axes = axes[:, None]

    for v in range(V):
        # Original images
        ax = axes[0, v*2]
        ax.imshow(images[v])
        ax.set_title(f"View {v} (orig)")

        # Augmented images
        ax = axes[0, v*2 + 1]
        ax.imshow(images_aug[v])
        ax.set_title(f"View {v} (aug)")

        # Mesh projections
        for r, (name, verts) in enumerate(meshes.items(), start=1):
            # Original
            ax = axes[r, v*2]
            ax.imshow(images[v])
            proj = project_points(verts, intrinsics[v], extrinsics[v])
            ax.scatter(proj[:, 0]-.5, proj[:, 1]-.5, s=1, label=name)
            ax.set_title(f"{name} proj (orig v{v})")

            # Augmented
            ax = axes[r, v*2 + 1]
            ax.imshow(images_aug[v])
            proj = project_points(verts, intrinsics_aug[v], extrinsics[v])
            ax.scatter(proj[:, 0]-.5, proj[:, 1]-.5, s=1, label=name)
            ax.set_title(f"{name} proj (aug v{v})")

    plt.tight_layout()
    return fig
