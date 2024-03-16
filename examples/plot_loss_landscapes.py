import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import math

from utils import lame_from_young

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


def get_loss_first_and_last(x, y) -> float:
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
        positions = torch.stack([positions[0], positions[-1]])
        positions_gt = torch.stack([positions_pseudo_gt[0], positions_pseudo_gt[positions.shape[0] - 1]])
        loss = mse(positions, positions_gt)
    return loss.item()


def get_loss_weighted(x, y) -> float:
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
        weights = torch.zeros(training_frame_count)
        # weights[0] = 2
        # weights[-1] = 2
        weights[12:16] = 1
        # weights[0] = 1
        # weights[24] = 1
        loss = (weights * torch.sum(torch.mean((positions - positions_pseudo_gt)**2, dim=1), dim=1)).sum()
    return loss.item()


def get_loss_young(E, nu) -> float:
    global progress_counter
    print(f"{progress_counter}/{total_iterations}")
    progress_counter += 1
    k_mu, k_lambda = lame_from_young(E, nu)
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                          scale, velocity, points, tet_indices, density,
                                          k_mu, k_lambda, k_damp, training_sim_steps,
                                          sim_dt, render_steps)
        save_positions(positions, f"{output_dir}/positions_pred.npz")
        loss = mse(positions, positions_pseudo_gt[:training_frame_count])
    return loss.item()


vectorized_lame = np.vectorize(get_loss)
vectorized_young = np.vectorize(get_loss_young)
vectorized_first_last = np.vectorize(get_loss_first_and_last)
vectorized_weighted = np.vectorize(get_loss_weighted)


def plot_mu_loss_landscape(load=False):
    steps = 100
    center = 1e4
    radius = 3500

    low_bound = center - radius
    high_bound = center + radius

    xs = torch.linspace(low_bound, high_bound, steps=steps)
    ys = torch.ones_like(xs) * 1e4

    if not load:
        zs = vectorized_lame(xs, ys)
        np.savez(f"{output_dir}/mu_loss_landscape.npz", zs)
    else:
        zs = np.load(f"{output_dir}/mu_loss_landscape.npz")["arr_0"]

    plt.plot(xs, zs)
    plt.xlabel("$\mu$")
    plt.ylabel("Loss")
    plt.show()
def plot_lambda_loss_landscape(load=False):
    steps = 100
    center = 1e4
    radius = 3500

    low_bound = center - radius
    high_bound = center + radius

    xs = torch.linspace(low_bound, high_bound, steps=steps)
    ys = torch.ones_like(xs) * 1e4

    if not load:
        zs = vectorized_lame(ys, xs)
        np.savez(f"{output_dir}/lambda_loss_landscape.npz", zs)
    else:
        zs = np.load(f"{output_dir}/lambda_loss_landscape.npz")["arr_0"]

    plt.plot(xs, zs)
    plt.xlabel("$\lambda$")
    plt.ylabel("Loss")
    plt.show()


def plot_joined_loss_landscape(output_filename, function=vectorized_lame, load=False, steps=100, center=1e4, radius=3500):
    low_bound = center - radius
    high_bound = center + radius

    xs = torch.linspace(low_bound, high_bound, steps=steps)
    ys = torch.linspace(low_bound, high_bound, steps=steps)
    global total_iterations
    total_iterations = len(xs) * len(ys)
    X, Y = np.meshgrid(xs, ys)
    if not load:
        Z = function(X, Y)
        np.savez(output_filename, Z)
    else:
        Z = np.load(output_filename)["arr_0"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Loss')

    fig.colorbar(p, ax=ax)

    plt.show()


def plot_joined_loss_first_and_last_landscape(load=False):
    print("Loss first and last frame")
    steps = 5
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
        Z = vectorized_first_last(X, Y)
        np.savez(f"{output_dir}/joined_loss_landscape_first_last_{steps}.npz", Z)
    else:
        Z = np.load(f"{output_dir}/joined_loss_landscape_first_last_{steps}.npz")["arr_0"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Loss')

    fig.colorbar(p, ax=ax)

    plt.show()


def plot_joined_loss_weighted_landscape(load=False):
    print("Loss weighted frames")
    steps = 5
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
        Z = vectorized_weighted(X, Y)
        np.savez(f"{output_dir}/joined_loss_landscape_weighted_{steps}.npz", Z)
    else:
        Z = np.load(f"{output_dir}/joined_loss_landscape_weighted_{steps}.npz")["arr_0"]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    p = ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Loss')

    fig.colorbar(p, ax=ax)

    plt.show()


def E_from_mu_lambda(mu, lam):
    return 2 * (1 + (lam / (2 * mu + 2 * lam))) * mu


def nu_from_mu_lambda(mu, lam):
    return lam / (2 * mu + 2 * lam)


def plot_joined_loss_landscape_young(load=False):
    steps = 4
    center = 1e4

    center_E = E_from_mu_lambda(center, center)
    center_nu = nu_from_mu_lambda(center, center)

    radius = 3500
    radius_E = E_from_mu_lambda(radius, radius)
    radius_nu = nu_from_mu_lambda(radius, radius)

    xs = torch.linspace(center_E - radius_E, center_E + radius_E, steps=steps)
    ys = torch.linspace(center_nu - radius_nu, center_nu + radius_nu, steps=steps)
    global total_iterations
    total_iterations = len(xs) * len(ys)
    X, Y = np.meshgrid(xs, ys)
    if not load:
        Z = vectorized_young(X, Y)
        np.savez(f"{output_dir}/joined_loss_landscape_young_{steps}.npz", Z)
    else:
        Z = np.load(f"{output_dir}/joined_loss_landscape_young_{steps}.npz")["arr_0"]

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
    #TODO: Clean this mess up.
    print(f"Loss at known optimum: {loss}")
    output_filename = f"{output_dir}/joined_loss_landscape_12-26.npz"
    plot_joined_loss_landscape(output_filename, load=False, steps=10, function=vectorized_weighted)
