from pathlib import Path
from pca_gauss import get_gtempeh_origin_offset, gaussian_locations_2_mesh, template_triangle_uvs, template_triangles, template_verts, gtempeh_points_2_gem_points, build_dataset, seed_everything, DatasetMode, vis_3d_point_clouds
from omegaconf import OmegaConf
import torch
import sys
import einops
import numpy as np
import trimesh
import pudb

def closest_points_on_mesh(mesh: trimesh.Trimesh, points: np.ndarray):
    """
    For each point in a point cloud, find the closest point on a mesh,
    and return barycentric coordinates, vertex indices, and distances.

    Args:
        mesh: trimesh.Trimesh object
        points: (N, 3) numpy array of query points

    Returns:
        barycentric_coords: (N, 3) array of barycentric coordinates
        vertex_indices: (N, 3) array of vertex indices for the triangle
        distances: (N,) array of distances to the closest point
        closest_points: (N, 3) array of closest surface points
    """
    # Find nearest points on mesh surface
    closest_points, distances, face_indices = trimesh.proximity.closest_point(mesh, points)

    # Get the triangle vertices for each face
    triangles = mesh.triangles[face_indices]  # (N, 3, 3)

    # Compute barycentric coordinates for the closest points
    barycentric_coords = trimesh.triangles.points_to_barycentric(triangles, closest_points)  # (N, 3)

    # Get the corresponding vertex indices for each face
    vertex_indices = mesh.faces[face_indices]  # (N, 3)

    return barycentric_coords, vertex_indices, distances, closest_points


def main(config):
    alignment_data_idx = 22
    outpath_ava2flame = 'assets/meshes/mapping_ava2flame.npz'
    outpath_flame2gaussians = 'assets/meshes/mapping_flame2gaussians.npz'
    exclude_faces = [# excluding connected lip faces
        11314, 11380, 11381, 11382, 11383, 11384, 11385, 11386, 11387, 11388, 11389, 11390, 11391, 11392, 11393, 11394, 11395, 11396, 11397, 11398, 11399, 11400, 11401, 11402, 11403, 11404, 11405, 11406, 11407, 11408, 11409, 11410, 11411, 11412, 11413, 11414, 11415, 11416, 11417, 11418, 11419, 11420, 11421, 11422, 11423, 11424, 11425, 11426, 11427, 11428, 11429, 11430, 11431]
    dataset = build_dataset(config, camera_list=[config.data.test_camera], mode=DatasetMode.validation)
    single = dataset[alignment_data_idx]
    for k, v in single.items():
        if isinstance(v, np.ndarray):
            v = torch.tensor(v, device='cuda')
        single[k] = v

    frame_idx = int(Path(single['image_path']).name.split('_')[0])
    gtempeh_gaussians_path = sorted([p for p in Path(f'{config.train.gtempeh_path}/{single["exp"]}').iterdir() if p.name.isnumeric()])[frame_idx] / 'gaussians.pt'
    gtempeh_cameras_path = gtempeh_gaussians_path.parent/'cameras.pt'
    gtempeh_cameras = torch.load(gtempeh_cameras_path)['C2W']
    gtempeh_origin_offset = get_gtempeh_origin_offset(gtempeh_cameras=gtempeh_cameras, single=single)
    gtempeh_gaussians = torch.load(gtempeh_gaussians_path)
    ava_verts = gaussian_locations_2_mesh(gtempeh_gaussians['xyz'][None], triangle_uvs=template_triangle_uvs, triangle_vert_ids=template_triangles, num_verts=len(template_verts))[0]
    ava_verts = gtempeh_points_2_gem_points(ava_verts, origin_offset = gtempeh_origin_offset)
    V, C, H, W = gtempeh_gaussians['xyz'].shape
    xyz = einops.rearrange(gtempeh_gaussians['xyz'], 'v c h w -> (v h w) c')
    xyz = gtempeh_points_2_gem_points(xyz, origin_offset = gtempeh_origin_offset)

    flame_verts = single['geom_vertices']
    flame_faces = single['geom_faces']
    flame_verts = einops.einsum(single['root_RT'].float(), torch.cat([single['geom_vertices'].float(), torch.ones_like(single['geom_vertices'].float()[:, :1])], dim=-1), 'i j, n j -> n i')[:,:3]
    
    # for each flame vertex find closest point on ava mesh -> get face, barycentrics, and distance
    ava_faces = template_triangles.cpu().clone()
    face_mask = torch.ones(len(ava_faces), dtype=torch.bool)
    face_mask[exclude_faces]=False
    ava_faces = ava_faces[face_mask]
    ava_mesh = trimesh.Trimesh(vertices=ava_verts.cpu(), faces=ava_faces, process=False)
    flame_mesh = trimesh.Trimesh(vertices=flame_verts.cpu(), faces=flame_faces.cpu(), process=False)

    # ava2flame
    bary_coords, vertex_idcs, distances, closest_pts = closest_points_on_mesh(ava_mesh, flame_verts.cpu().numpy())
    Path(outpath_ava2flame).parent.mkdir(exist_ok=True, parents=True)
    np.savez(outpath_ava2flame, barycentric_coordinates=bary_coords, vertex_indices=vertex_idcs, distances=distances)
    print(f'Stored mapping to {outpath_ava2flame}')

    # visualizing verts
    vis_3d_point_clouds(dict(ava=ava_verts.cpu().numpy(), flame=flame_verts.cpu().numpy(), closest=closest_pts), 'demos/mapping_ava2flame_vis.html')

    # flame2gaussians
    bary_coords, vertex_idcs, distances, closest_pts = closest_points_on_mesh(flame_mesh, xyz.cpu().numpy())
    Path(outpath_flame2gaussians).parent.mkdir(exist_ok=True, parents=True)
    bary_coords = einops.rearrange(bary_coords, '(v h w) c -> v c h w', v=V, h=H, w=W)
    vertex_idcs = einops.rearrange(vertex_idcs, '(v h w) c -> v c h w', v=V, h=H, w=W)
    distances = einops.rearrange(distances, '(v c h w) -> v c h w', v=V, h=H, w=W)
    np.savez(outpath_flame2gaussians, barycentric_coordinates=bary_coords, vertex_indices=vertex_idcs, distances=distances)
    print(f'Stored mapping to {outpath_flame2gaussians}')

    # visualizing verts
    vis_3d_point_clouds(dict(gaussians=xyz.cpu().numpy(), flame=flame_verts.cpu().numpy(), closest=closest_pts), 'demos/mapping_flame2gaussians_vis.html')


if __name__ == "__main__":
    # python get_ava2flame_mapping.py configs/distillation/INQ807_TEETHFOCUS/gtempeh_pca_all.yml
    path = sys.argv[1]
    config = OmegaConf.load(path)
    main(config)
