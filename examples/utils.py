import numpy as np
import pyvista as pv
import tetgen
import torch

from kaolin.io.obj import import_mesh


def export_obj(vertices, faces, filepath):
    """
    Export vertices and faces to a Wavefront OBJ file.

    Args:
    - vertices (list of tuples): List of vertex coordinates, each tuple representing (x, y, z) coordinates.
    - faces (list of tuples): List of faces, where each tuple contains the indices of vertices forming a face.
    - filepath (str): Filepath to save the OBJ file.
    """
    with open(filepath, 'w') as f:
        # Write vertices
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(vertex[0], vertex[1], vertex[2]))

        # Write faces
        for face in faces:
            # Increment indices by 1 as OBJ format indices start from 1
            f.write('f {} {} {}\n'.format(face[0] + 1, face[1] + 1, face[2] + 1))


def read_tet_mesh(filepath):
    vertices = []
    faces = []
    with open(filepath, "r") as mesh:
        for line in mesh:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == "v":
                vertices.append([float(d) for d in data[1:]])
            elif data[0] == "t":
                faces.append([int(d) for d in data[1:]])
    vertices = [tuple(v) for v in vertices]
    vertices = np.asarray(vertices).astype(np.float32)
    faces = [f for face in faces for f in face]
    return vertices, faces


def tetrahedralize(vertices, faces, order=1, mindihedral=5, minratio=20):
    # Not the cleanest, but fastest to implement
    verts = vertices.clone().detach().numpy()
    faces = faces.clone().detach().numpy()
    pv_mesh = pv.make_tri_mesh(verts, faces=faces)
    tet = tetgen.TetGen(pv_mesh)
    tet.make_manifold()
    tet.tetrahedralize()
    faces = tet.grid.cells_dict[pv.CellType.TETRA]
    faces = [f for face in faces for f in face]
    return np.asarray(tet.grid.points, dtype=np.float32).astype(np.float32), faces


def load_pseudo_gt_mesh(path: str):
    path_to_mesh_data = f"{path}/filtered_meshes"
    # from the obj we will keep the faces
    mesh = import_mesh(f"{path_to_mesh_data}/filtered_mesh_0.obj")
    # but load vertex positions from the registered source data
    verts = torch.tensor(np.loadtxt(f"{path_to_mesh_data}/registered_vertices/registered_source_0.txt", delimiter="\t"))
    return tetrahedralize(verts, mesh.faces)


def load_mesh(path: str, load_from_gaussians=False):
    if path.endswith(".tet"):
        points, tet_indices = read_tet_mesh(path)
    else:
        mesh = import_mesh(path)
        points, tet_indices = tetrahedralize(mesh.vertices, mesh.faces)
    return points, tet_indices
