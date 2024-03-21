from utils import load_mesh
import numpy as np

if __name__ == '__main__':
    path_to_mesh = "/media/pavlos/One Touch/datasets/gt_generation/magic-salad/training_mesh.obj"
    base_output_path = "/media/pavlos/One Touch/datasets/gt_generation/magic-salad/tetrahedrals"

    points, tet_indices = load_mesh(path_to_mesh)

    vertex_path = f"{base_output_path}/verts.txt"
    np.savetxt(vertex_path, points, delimiter=",")

    tet_indices = np.array(tet_indices, dtype=int)
    tet_index_path = f"{base_output_path}/tets.txt"
    np.savetxt(tet_index_path, tet_indices, fmt="%d")
