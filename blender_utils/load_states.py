import bpy
import os
import numpy as np

# INSTRUCTIONS
# 0) This script should be run from inside Blender.
# 1) Set the paths below to point to where the simulation mesh and the numpy states are saved.
DATA_PATH = "/output"
MESH_PATH = f"{DATA_PATH}/simulation_mesh.obj"
GT_PATH = f"{DATA_PATH}/positions_gt.npz"
UNOPTIMIZED_PATH = f"{DATA_PATH}/unoptimized.npz"
PREDICTED_PATH = f"{DATA_PATH}/predicted.npz"
# 2) Run

def add_point_cloud_from_numpy(vertices, name="Simulated Mesh"):
    # Create a new mesh and object
    mesh = bpy.data.meshes.new(name=name)
    obj = bpy.data.objects.new(name=name, object_data=mesh)

    # Link the object to the scene
    bpy.context.scene.collection.objects.link(obj)

    # Create mesh data
    mesh.from_pydata(vertices, [], [])

    # Update mesh geometry and bounding box
    mesh.update()

    return obj


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
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj_object.name = obj_name
        print("Object created successfully.")
    else:
        print("No mesh objects found in the OBJ file.")
    return obj_object


def animate(obj, points: np.ndarray):
    mesh = obj.data
    for frame, p in enumerate(points):
        for i, vert in enumerate(mesh.vertices):
            mesh.vertices[i].co = p[i]
            mesh.vertices[i].keyframe_insert(data_path="co", frame=frame)
        mesh.update()


# Create ground truth
gt_object = create_object_from_obj(MESH_PATH, obj_name="Ground Truth")
gt_points = np.load(GT_PATH)["arr_0"]
animate(gt_object, gt_points)

# Create unoptimized
unoptimized_object = create_object_from_obj(MESH_PATH, obj_name="Unoptimized")
unoptimized_points = np.load(UNOPTIMIZED_PATH)["arr_0"]
animate(unoptimized_object, unoptimized_points)

# Create predicted object
predicted_object = create_object_from_obj(MESH_PATH, obj_name="Predicted")
predicted_points = np.load(PREDICTED_PATH)["arr_0"]
animate(predicted_object, predicted_points)
