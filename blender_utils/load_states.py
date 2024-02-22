import bpy
import os
import numpy as np

from mathutils import Vector
from scipy.spatial import KDTree

DATA_PATH = "examples/output"
MESH_PATH = f"{DATA_PATH}/simulation_mesh.obj"
GT_PATH = f"{DATA_PATH}/positions_gt.npz"
UNOPTIMIZED_PATH = f"{DATA_PATH}/unoptimized.npz"
PREDICTED_PATH = f"{DATA_PATH}/predicted.npz"


def create_point_cloud(vertices, matrix):
    """
    Creates a point cloud in Blender from a NumPy array of vertices.

    Parameters:
    - vertices (numpy.ndarray): Array of shape (N, 3) containing vertex coordinates.
    """
    # Create an empty mesh
    mesh = bpy.data.meshes.new("PointCloudMesh")
    obj = bpy.data.objects.new("PointCloud", mesh)
    bpy.context.scene.collection.objects.link(obj)

    vertices = [matrix @ Vector(vert) for vert in vertices]

    # Create vertices
    mesh.from_pydata(vertices, [], [])

    # Update mesh geometry
    mesh.update()


def create_object_from_obj(obj_file_path, obj_name="Simulation Object"):
    """
    Creates an object in Blender from an OBJ file.

    Parameters:
    - obj_file_path (str): Path to the OBJ file.
    """
    # Check if the file exists
    if not os.path.isfile(obj_file_path):
        print("Error: File not found.")
        return

    # Import OBJ file
    bpy.ops.wm.obj_import(filepath=obj_file_path)

    # Select the newly imported object
    obj_object = bpy.context.selected_objects[0]
    if obj_object:
        bpy.context.view_layer.objects.active = obj_object
        obj_object.name = obj_name
        print("Object created successfully.")
    else:
        print("No mesh objects found in the OBJ file.")
    return obj_object


def animate(obj, points: np.ndarray, index_map):
    mesh = obj.data
    for frame, p in enumerate(points):
        for i, vert in enumerate(mesh.vertices):
            mesh.vertices[i].co = p[index_map[i]]
            mesh.vertices[i].keyframe_insert(data_path="co", frame=frame)
        mesh.update()


# Create ground truth
gt_object = create_object_from_obj(MESH_PATH, obj_name="Ground Truth")
gt_points = np.load(GT_PATH)["arr_0"]

# Importing seems to break the index correspondences between the mesh and positions
# so we recalculate them based on nearest distance (in object space)
first_frame = gt_points[0]
kdtree = KDTree(first_frame)
index_map = {}
for vert_ix, vert in enumerate(gt_object.data.vertices):
    dist, ix = kdtree.query(np.array(vert.co))
    index_map[vert_ix] = ix

animate(gt_object, gt_points, index_map)

matrix = gt_object.matrix_world

# Create unoptimized
unoptimized_object = create_object_from_obj(MESH_PATH, obj_name="Unoptimized")
unoptimized_points = np.load(UNOPTIMIZED_PATH)["arr_0"]
animate(unoptimized_object, unoptimized_points)

# Create predicted object
predicted_object = create_object_from_obj(MESH_PATH, obj_name="Predicted")
predicted_points = np.load(PREDICTED_PATH)["arr_0"]
animate(predicted_object, predicted_points)
