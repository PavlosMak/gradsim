import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import math

from examples.utils import load_mesh, save_positions
from gradsim import dflex as df

from examples.training_utils import load_gt_positions, forward_pass

output_dir = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/output"

path_to_config = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/sampledata/configs/thinner_torus.json"

with open(path_to_config) as config_file:
    simulation_config = json.load(config_file)

training_config = simulation_config["training"]

positions_pseudo_gt = load_gt_positions(training_config)
save_positions(positions_pseudo_gt, f"{output_dir}/positions_pseudo_gt.npz")

position = tuple((0, 0, 0))
velocity = tuple(simulation_config["initial_velocity"])
scale = simulation_config["scale"]
density = simulation_config["density"]
k_mu = simulation_config["mu"]
k_lambda = simulation_config["lambda"]
k_damp = simulation_config["damp"]

r2 = eval(training_config["sim_mesh_rotation"])
sim_scale = training_config["sim_scale"]
sim_scale = 1.0

points, tet_indices = load_mesh(training_config["training_mesh"])
points = sim_scale * (df.quat_to_matrix(r2) @ points.transpose(1, 0)).transpose(1, 0)

training_frame_count = training_config["frame_count"]
sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
training_sim_steps = int(training_sim_duration / sim_dt)

render_steps = simulation_config["sim_substeps"]

mse = torch.nn.MSELoss(reduction="sum")

total_iterations = 0
progress_counter = 0


def get_loss(x, y) -> float:
    global progress_counter
    print(f"{progress_counter}/{total_iterations}")
    progress_counter += 1
    k_mu = x
    k_lambda = y
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                          scale, velocity, points, tet_indices, density,
                                          k_mu, k_lambda, k_damp, training_sim_steps,
                                          sim_dt, render_steps)
        save_positions(positions, f"{output_dir}/positions_pred.npz")
        loss = mse(positions, positions_pseudo_gt[:training_frame_count])
    return loss.item()


vectorized_fun = np.vectorize(get_loss)


def plot_mu_loss_landscape():
    # TODO:
    pass


def plot_lambda_loss_landscape():
    # TODO: Implement Lambda loss landscape
    pass


def plot_joined_loss_landscape(load=False):
    steps = 100
    center = 1e4
    radius = 3500
    low_bound = center - radius
    high_bound = center + radius

    xs = torch.linspace(low_bound, high_bound, steps=steps)
    ys = torch.linspace(low_bound, high_bound, steps=steps)
    global total_iterations
    total_iterations = len(xs) * len(ys)
    X, Y = np.meshgrid(xs, ys)
    if not load:
        Z = vectorized_fun(X, Y)
        np.savez(f"{output_dir}/joined_loss_landscape.npz", Z)
    else:
        Z = np.load(f"{output_dir}/joined_loss_landscape.npz")["arr_0"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Loss')

    fig.colorbar(p, ax=ax)

    plt.show()


if __name__ == '__main__':
    loss = get_loss(1e4, 1e4)
    print(f"Loss at known optimum: {loss}")
    plot_joined_loss_landscape(load=False)
