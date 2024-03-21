import numpy as np
import pyvista as pv
import tetgen
import torch

from kaolin.io.obj import import_mesh
import matplotlib.pyplot as plt

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
    # tet.tetrahedralize(mindihedral=5, minratio=20)
    tet.tetrahedralize(steinerleft=-1)
    # tet.tetrahedralize()
    faces = tet.grid.cells_dict[pv.CellType.TETRA]
    faces = [f for face in faces for f in face]
    surface_points = np.asarray(tet.grid.points, dtype=np.float32).astype(np.float32)[:len(verts)]
    return np.asarray(tet.grid.points, dtype=np.float32).astype(np.float32), faces, surface_points


def load_tet_directory(path: str):
    verts = np.loadtxt(f"{path}/verts.txt", delimiter=",")
    tet_indices = np.loadtxt(f"{path}/tets.txt").astype(np.int32)
    return verts, tet_indices

def load_pseudo_gt_mesh(path: str):
    path_to_mesh_data = f"{path}/filtered_meshes"
    # from the obj we will keep the faces
    mesh = import_mesh(f"{path_to_mesh_data}/filtered_mesh_0.obj")
    # but load vertex positions from the registered source data
    verts = torch.tensor(np.loadtxt(f"{path_to_mesh_data}/registered_vertices/registered_source_0.txt", delimiter="\t"))
    return tetrahedralize(verts, mesh.faces)


def load_mesh(path: str, load_from_gaussians=False):
    surface_points = None #TODO: TERRIBLE, MAKE BETTER
    if path.endswith(".tet"):
        points, tet_indices = read_tet_mesh(path)
    else:
        mesh = import_mesh(path)
        points, tet_indices, surface_points = tetrahedralize(mesh.vertices, mesh.faces)
    return points, tet_indices, surface_points


def get_tet_volume(v0, v1, v2, v3):
    v1v0 = v1 - v0
    v2v0 = v2 - v0
    v3v0 = v3 - v0
    return 0.1666 * torch.dot(torch.linalg.cross(v1v0, v2v0), v3v0)


def get_volumes(tet_indices, vertex_buffer):
    tet_volumes = []
    index = 0
    while index < len(tet_indices):
        i0 = tet_indices[index]
        i1 = tet_indices[index + 1]
        i2 = tet_indices[index + 2]
        i3 = tet_indices[index + 3]
        index += 4
        tet_volumes.append(get_tet_volume(vertex_buffer[i0], vertex_buffer[i1], vertex_buffer[i2], vertex_buffer[i3]))
    return torch.stack(tet_volumes)


def lame_from_young(E: float, nu: float):
    """
    Calculate the Lame parameters from Young's modulus
    and Poisson ratio
    Args:
        E: Young's modulus
        nu: Poisson ratio
    Returns: (mu, lambda)
    """
    mu = E / (2 * (1 + nu))
    lam = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    return mu, lam


def young_from_lame(mu, lam):
    E = 2 * (1 + (lam / (2 * mu + 2 * lam))) * mu
    nu = lam / (2 * mu + 2 * lam)
    return E, nu


def get_ground_truth_lame(simulation_config: dict) -> (float, float):
    if "E" in simulation_config["training"] and "nu" in simulation_config["training"]:
        training_config = simulation_config["training"]
        return lame_from_young(training_config["E"], training_config["nu"])
    return simulation_config["mu"], simulation_config["lambda"]


def get_ground_truth_young(simulation_config: dict) -> (float, float):
    if "E" in simulation_config["training"] and "nu" in simulation_config["training"]:
        training_config = simulation_config["training"]
        return training_config["E"], training_config["nu"]
    else:
        return young_from_lame(simulation_config["mu"], simulation_config["lambda"])


def save_positions(positions: torch.Tensor, filename: str) -> None:
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(filename, positions_np)

def plot_histogram(data, bins=10, xlabel="Values", ylabel="Frequency", title="Histogram"):
    """
    Plots a histogram from a list of data.

    Parameters:
    - data: List of numerical data.
    - bins: Number of bins for the histogram. Default is 10.
    - xlabel: Label for the x-axis. Default is "Values".
    - ylabel: Label for the y-axis. Default is "Frequency".
    - title: Title of the histogram plot. Default is "Histogram".
    """
    plt.hist(data, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()