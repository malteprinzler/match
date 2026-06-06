from typing import Tuple
import numpy as np
from typing import Union, TextIO, List
import json
import numpy as np
import numpy.typing as npt
import scipy
from match.utils import file_util, general_util
import tensorflow as tf
import torch
import trimesh
from typing import Union, TextIO, Dict, List
ObjectType = Dict[str, Union[List[np.ndarray], np.ndarray]]

_INT = np.int32
_FLOAT = np.float32
_IntArray = npt.NDArray[_INT]
_FloatArray = npt.NDArray[_FLOAT]


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
    vertex_group_names: list[str],
    group: str | None = None,
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


# GOOGLE CODE

def compute_vertex_by_face_matrix(
    *, num_vertices: int, faces: torch.Tensor
) -> torch.Tensor:
  """Computes the mapping of vertex (rows) to face (columns) indices.

  Args:
    num_vertices: The number of vertices in the mesh.
    faces: The mesh triangles, (F, 3).

  Returns:
    The (num_vertices, num_faces)-dimensional mapping matrix. The returned
    matrix is a sparse tensor where the i-the row represents the i-th vertex and
    the j-th column represents the j-th face. The value of the matrix at (i, j)
    is 1.0 if the vertex i is part of the face j, and 0 otherwise.
  """
  num_faces = faces.shape[0]
  faces = general_util.torch_to_numpy(faces)
  row = faces.flatten()
  col = np.array([range(faces.shape[0])] * 3).T.flatten()
  indices = np.stack((row, col)).tolist()
  data = [1.0] * col.shape[0]
  return torch.sparse_coo_tensor(
      indices=indices,
      values=data,
      size=(num_vertices, num_faces),
      dtype=torch.float32,
      device=faces.device,
  )


def compute_vertex_normals(
    *,
    vertices: torch.Tensor,
    faces: torch.Tensor,
    vertex_by_face_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
  """Compute the normalized vertex normals.

  Normal computation by averaging the face normals adjacent to the vertex.
  Args:
      vertices: The batched vertex tensors, (B, N, 3).
      faces: The mesh triangles shared for all vertex tensors the batch, (F, 3).
      vertex_by_face_matrix: The mapping between vertex indices (rows) and face
        indices (columns). If None, it will be computed.

  Returns:
      vertex_normals: The batched normalized vertex normals, (B, N, 3).
  """
  batch_size, num_vertices = vertices.shape[:2]
  device = vertices.device

  faces = faces.to(device)
  vertices1 = torch.index_select(vertices, dim=1, index=faces[:, 0])
  vertices2 = torch.index_select(vertices, dim=1, index=faces[:, 1])
  vertices3 = torch.index_select(vertices, dim=1, index=faces[:, 2])

  # Edges of all triangles.
  edges1 = vertices2 - vertices1
  edges2 = vertices3 - vertices1

  # Compute face normals, (B, F, 3).
  face_normals = torch.cross(edges1, edges2, dim=-1)

  # Mapping of vertices to faces.
  if vertex_by_face_matrix is None:
    vertex_by_face_matrix = compute_vertex_by_face_matrix(
        num_vertices=num_vertices, faces=faces
    )
  vertex_by_face_matrix = vertex_by_face_matrix.to(device)

  # Compute vertex normals by averaging the unormalized normals of all faces
  # adjacent to the vertex.
  face_normals = face_normals.permute(1, 0, 2)  # (F, B, 3).
  face_normals = face_normals.reshape([-1, batch_size * 3])  # (F, B*3).
  vertex_normals = torch.matmul(vertex_by_face_matrix, face_normals)  # (N, B*3)
  vertex_normals = vertex_normals.view([num_vertices, batch_size, 3])
  vertex_normals = vertex_normals.permute(1, 0, 2)  # (B, N, 3).
  return torch.nn.functional.normalize(vertex_normals, dim=-1)


class Mesh:
  """Mesh data container."""

  def __init__(
      self,
      vertices: _FloatArray | None = None,
      faces: _IntArray | None = None,
      faceuvcoords: _IntArray | None = None,
      file_path: file_util.Path | None = None,
  ):
    """Initialize a mesh from vertices and faces, or load it from file.

    If file_path is provided, vertices and faces will be ignored. Initiating a
    mesh assumes that either vertices or a file_path is provided. When loading
    a mesh from a JSON file, it must contain a field for "vertices" and
    optionally one for "faces".

    Args:
      vertices: A list of vertices, (N, 3).
      faces (optional): A list of face indices, (F, 3).
      file_path: The path of a mesh file in OBJ, PLY, or JSON file format.
    """
    if file_path:
      self.vertices, self.faces, self.faceuvcoords = self._load(file_path)
    elif vertices is not None:
      self.vertices = np.copy(vertices)
      self.faces = np.copy(faces) if faces is not None else None
      self.faceuvcoords = np.copy(faceuvcoords) if faceuvcoords is not None else None
    else:
      raise ValueError("Either vertices or file_path must be specified.")

  def sample_surface(self, vertex_count: int) -> _FloatArray:
    """Sample points in the mesh surface.

    Args:
      vertex_count: The number of vertices to be sampled in the mesh surface.

    Returns:
      Sampled surface points, (vertex_count, 3).
    """
    if self.faces is None:
      raise ValueError("Mesh has no faces which are required for sampling.")
    mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
    return trimesh.sample.sample_surface(mesh, vertex_count)[0]

  def subdivide(self):
    """Subdivide the mesh, where each face is replaced by four smaller faces."""
    if self.faces is None:
      raise ValueError("Mesh has no faces which are required for subdivision.")
    mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
    subdivided_mesh = mesh.subdivide()
    self.vertices = subdivided_mesh.vertices
    self.faces = subdivided_mesh.faces

  def remove_vertices(self, vertex_indices: _IntArray):
    """Removes all specified indices from the mesh.

    Args:
      vertex_indices: An array with the indices of the vertices to be removed.
    """
    if self.faces is None:
      keep_vertex_indices = np.setdiff1d(
          np.arange(self.vertices.shape[0]), vertex_indices
      )
      self.vertices = self.vertices[keep_vertex_indices]
    else:
      # Create a binary vertex mask with True, if a vertex is to be removed, and
      # False otherwise.
      vertex_mask = np.isin(np.arange(self.vertices.shape[0]), vertex_indices)
      # A face is to be removed, if any of its vertices is to be removed.
      face_mask = vertex_mask[self.faces].any(axis=1)
      # Update the vertex mask to only include vertices that are part of at
      # least one face that is kept after masking.
      keep_face_mask = np.invert(face_mask)
      keep_face_mask_ids = np.where(keep_face_mask)[0]
      keep_vertex_mask_ids = np.unique(self.faces[keep_face_mask_ids].ravel())
      keep_vertex_mask = np.isin(
          np.arange(self.vertices.shape[0]), keep_vertex_mask_ids
      )
      # Mask the vertices and faces.
      mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
      mesh.update_faces(keep_face_mask)
      mesh.update_vertices(keep_vertex_mask)
      self.vertices = mesh.vertices
      self.faces = mesh.faces

  def boundary_vertex_indices(self) -> _IntArray:
    """Computes the vertex indices of the mesh boundaries.

    Boundary vertices are defined as vertices of edges shared only by one face.

    Returns:
      The vertex indices of the mesh boundaries.
    """
    num_vertices = self.vertices.shape[0]
    if self.faces is None:
      raise ValueError(
          "Mesh has no faces which are required for determining the mesh"
          " boundary."
      )
    polygon_dim = self.faces.shape[-1]
    mtx_fused = scipy.sparse.csr_matrix((num_vertices, num_vertices))
    for i in range(0, polygon_dim):
      indices_a = self.faces[:, i]
      indices_b = self.faces[:, (i + 1) % polygon_dim]
      ij = np.vstack((indices_a.reshape(1, -1), indices_b.reshape(1, -1)))
      values = np.ones_like(indices_a)
      mtx = scipy.sparse.csr_matrix(
          (values, ij), shape=(num_vertices, num_vertices)
      )
      mtx_fused = mtx_fused + mtx + mtx.T
    mtx_fused = scipy.sparse.coo_matrix(mtx_fused)
    data = mtx_fused.data.reshape(-1, 1)
    boundary_edge_ids = np.where(data == 1)[0]
    boundary_vertex_ids = np.hstack(
        (mtx_fused.row[boundary_edge_ids], mtx_fused.col[boundary_edge_ids])
    )
    return np.unique(boundary_vertex_ids).astype(np.int32)

  def nearest_vertex_indices(self, query_points: _FloatArray) -> _IntArray:
    """Finds the nearest vertex to the query points."""
    mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
    _, v_closest_ids = mesh.nearest.vertex(query_points)
    return v_closest_ids.ravel().astype(np.int32)

  def save(self, file_path: file_util.Path):
    """Saves mesh as OBJ, PLY, or JSON file.

    Args:
      file_path: The path of the file to save the mesh to. The file type is
        determined by the file extension, e.g., .ply, .obj, or .json.
    """
    if file_path.suffix in [".ply", ".obj"]:
      file_type = file_path.suffix.strip(".").upper()
      if self.faces is not None:
        mesh = trimesh.Trimesh(self.vertices, self.faces, process=False)
      else:
        mesh = trimesh.Trimesh(self.vertices, process=False)
      with file_util.open_file(file_path, "wb") as f:
        mesh.export(f, file_type=file_type)
    elif file_path.suffix == ".json":
      data = {"vertices": self.vertices.tolist()}
      if self.faces is not None:
        data["faces"] = self.faces.tolist()
      with file_util.open_file(file_path, "wb") as f:
        json.dump(data, f)
    else:
      raise ValueError(f"Unsupported file type: {file_path.suffix}")

  def _load(
      self, file_path: file_util.Path
  ) -> tuple[_FloatArray, _IntArray | None, _FloatArray|None]:
    """Loads vertices and faces of an OBJ, PLY, or JSON file.

    Args:
      file_path: The path to an OBJ, PLY, or JSON geometry file.

    Returns:
      Vertices, faces, face uv coordinates of the loaded mesh.
    """
    vertices, faces, faces_uv, uv = None, None, None, None
    if file_path.suffix == '.obj':
       with file_util.open_file(file_path, 'r') as f:
        loaded_obj = load_obj(f)
       vertices = loaded_obj['v']
       uv = loaded_obj['vt']
       faces = loaded_obj['vi']
       faces_uv = loaded_obj['vti']
       

    elif file_path.suffix in [".ply"]:
      file_type = file_path.suffix.strip(".").upper()
      trimesh_load_kwargs = {
          "file_type": file_type,
          "force": "mesh",
          "process": False,
          "skip_materials": True,
          "maintain_order": True,
      }
      with file_util.open_file(file_path, "rb") as f:
        trimesh_mesh = trimesh.load(f, **trimesh_load_kwargs)
      vertices = np.array(trimesh_mesh.vertices).astype(_FLOAT)
      faces = (
          np.array(trimesh_mesh.faces).astype(_INT)
          if hasattr(trimesh_mesh, "faces")
          else None
      )
    elif file_path.suffix == ".json":
      with file_util.open_file(file_path, "r") as f:
        data = json.load(f)
      vertices = np.array(data["vertices"]).astype(_FLOAT)
      faces = np.array(data["faces"]).astype(_INT) if "faces" in data else None
    else:
      raise ValueError(f"Unsupported file type: {file_path.suffix}")
    
    if (uv is not None) and (faces_uv is not None):
       face_uv_coords = uv[faces_uv]
    else:
       face_uv_coords = None
    return vertices, faces, face_uv_coords


def load_obj(path: Union[str, TextIO], return_vn: bool = False) -> ObjectType:
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
