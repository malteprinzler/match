from typing import Tuple
import numpy as np
from typing import Union, TextIO, List, Optional

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

def remove_vertices_by_mask(vertices, unused_mask: np.ndarray):
    keep = np.nonzero(~unused_mask)[0]
    index_map = -np.ones(len(unused_mask), dtype=int)
    index_map[keep] = np.arange(len(keep))
    return vertices[keep]

def apply_universal_mapping(subject_verts: np.ndarray,
                            tri_vids: np.ndarray,
                            bary: np.ndarray,
                            nn: np.ndarray):
    V = subject_verts
    gathered = V[tri_vids] # (N, 3, 3)
    new_vertices = np.einsum('tvi, tv -> ti', gathered, bary)
    bad = np.isclose(bary.sum(axis=1), 0.0)
    if bad.any():
        new_vertices[bad] = V[nn[bad]]
    return new_vertices


def filter_mesh_by_vertex_group(
    faces: np.ndarray,
    vertex_group_weights: np.ndarray,
    vertex_group_names: List[str],
    group: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Filters a mesh by a vertex group.

  Args:
    faces: G-Nome triangles. np.ndarray of shape (F, 3)
    vertex_group_weights: Vertex group weights. np.ndarray of shape (G, V)
    vertex_group_names: Vertex group names. list of strings of length G
    group: Vertex group name. str

  Returns:
    face_mask: np.ndarray[bool] of shape (F,)
    vertex_mask: np.ndarray[bool] of shape (V,) including all vertices in faces
    for which at least one vertex is in the vertex group
  """
  if group is None:  # no filtering
    return np.ones(faces.shape[0], dtype=bool), np.ones(
        vertex_group_weights.shape[1], dtype=bool
    )
  else:
    group_idx = vertex_group_names.index(group)
    vertex_weights = vertex_group_weights[group_idx]
    filtered_vertex_idcs = np.where(vertex_weights > 0)[0]

    face_mask = np.any(np.isin(faces, filtered_vertex_idcs[:, None]), axis=1)
    faces_filtered = faces[face_mask]
    filtered_vert_idcs = np.unique(faces_filtered.flatten())
    vertex_mask = np.zeros((len(vertex_weights)), dtype=bool)
    vertex_mask[filtered_vert_idcs] = True

  return face_mask, vertex_mask


def get_uv_mesh(
    vertices: np.ndarray, faces: np.ndarray, face_uv_coords: np.ndarray
):
  """Every face gets individual vertices such that face uv coords can be expressed as vertex uv coords.

  Args:
    vertices: G-Nome vertices. np.ndarray of shape (B, V, 3)
    faces: G-Nome triangles. np.ndarray of shape (F, 3)
    face_uv_coords: UV coordinates of the G-Nome triangles. np.ndarray of shape
      (F, 3, 2)

  Returns:
    out_faces: np.ndarray of shape (F, 3)
    out_vertices: np.ndarray of shape (V, 3)
    out_vertex_uv_coords: np.ndarray of shape (V, 2)
  """

  out_faces = list()
  out_vertices = list()
  out_vertex_uv_coords = list()

  for f, f_uv in zip(faces, face_uv_coords):
    for vidx in range(3):
      out_vertices.append(vertices[:, f[vidx]])
      out_vertex_uv_coords.append(f_uv[vidx])
    out_faces.append(np.arange(len(out_vertices) - 3, len(out_vertices)))

  out_faces = np.stack(out_faces, axis=0)  # (F, 3), int
  out_vertices = np.stack(out_vertices, axis=1)  # (B, V, 3), float
  out_vertex_uv_coords = np.stack(out_vertex_uv_coords, axis=0)  # (V, 2), float
  return out_vertices, out_faces, out_vertex_uv_coords
