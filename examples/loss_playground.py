import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import math

from torch.nn import MSELoss

from examples.utils import load_mesh, save_positions, load_tet_directory
from gradsim import dflex as df

from examples.training_utils import load_gt_positions, forward_pass
from examples.losses import *

from pytorch3d.loss import chamfer_distance

output_dir = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/output"

path_to_config = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/sampledata/configs/thinner_torus_red.json"
# path_to_config = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/sampledata/configs/thinner_torus.json"


with open(path_to_config) as config_file:
    simulation_config = json.load(config_file)

training_config = simulation_config["training"]

positions_pseudo_gt = load_gt_positions(training_config)
# positions_pseudo_gt = positions_pseudo_gt + 0.1*torch.rand_like(positions_pseudo_gt)*positions_pseudo_gt
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
# points, tet_indices = load_mesh(training_config["training_mesh"])
points, tet_indices = load_tet_directory("/media/pavlos/One Touch/datasets/gt_generation/magic-salad/tetrahedrals")
points = sim_scale * (df.quat_to_matrix(r2) @ points.transpose(1, 0)).transpose(1, 0)

training_frame_count = training_config["frame_count"]
sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
training_sim_steps = int(training_sim_duration / sim_dt)

render_steps = simulation_config["sim_substeps"]

mse = torch.nn.MSELoss(reduction="sum")
fixedcorrloss = FixedCorrespondenceDistanceLoss(positions_pseudo_gt[0], points)
msecorrloss = MSECorrespondencesLoss(positions_pseudo_gt[0], points)
closest_loss = ClosestOnlyLoss(positions_pseudo_gt[0], points)

total_iterations = 0
progress_counter = 0

# mass_noise = torch.rand(592)
mass_noise = None


def get_loss(x, y) -> float:
    global progress_counter
    print(f"{progress_counter}/{total_iterations}")
    progress_counter += 1
    k_mu = x
    k_lambda = y
    with torch.no_grad():
        positions, model, _, _ = forward_pass(position, df.quat_identity(),
                                              1.0, velocity, points, tet_indices, density,
                                              k_mu, k_lambda, k_damp, training_sim_steps,
                                              sim_dt, render_steps, mass_noise=mass_noise)
        save_positions(positions, f"{output_dir}/positions_pred.npz")
        weights = torch.ones(training_frame_count)
        # weights[12:16] = 1
        C = 1
        # loss = (weights * torch.sum(torch.mean(C * (positions - positions_pseudo_gt) ** 2, dim=1), dim=1)).sum()
        # loss = torch.zeros(1)
        # loss = lossfn(positions, positions_pseudo_gt)
        loss = len(positions) * chamfer_distance(positions[17:20], positions_pseudo_gt[17:20])[0]
        # loss = msecorrloss(positions, positions_pseudo_gt)
        # loss = closest_loss(positions, positions_pseudo_gt)
    return loss.item()


def get_E(x, y) -> float:
    global progress_counter
    print(f"{progress_counter}/{total_iterations}")
    progress_counter += 1
    k_mu = x
    k_lambda = y
    E = k_mu * (3 * k_mu + 2 * k_lambda) / (k_lambda + k_mu)
    return E


def get_nu(x, y) -> float:
    global progress_counter
    print(f"{progress_counter}/{total_iterations}")
    progress_counter += 1
    k_mu = x
    k_lambda = y
    nu = k_lambda / (2 * (k_lambda + k_mu))
    return nu


vectorized_loss = np.vectorize(get_loss)
vectorized_E = np.vectorize(get_E)
vectorized_nu = np.vectorize(get_nu)

def plot_joined_loss_landscape(output_filename, function=vectorized_loss, load=False, steps=100, center_mu=1e4,
                               center_lambda=1e4,
                               radius_mu=3500, radius_lambda=3500, z_title="Loss"):
    global total_iterations
    xs = torch.linspace(center_mu - radius_mu, center_mu + radius_mu, steps=steps)
    ys = torch.linspace(center_lambda - radius_lambda, center_lambda + radius_lambda, steps=steps)
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
    ax.set_zlabel(z_title)

    fig.colorbar(p, ax=ax)

    plt.show()


def plot_lambda_loss_landscape(center_mu, center_lambda, output_file_name, radius=3500, steps=5, load=False,
                               function=vectorized_loss):
    global total_iterations

    low_bound = center_lambda - radius
    high_bound = center_lambda + radius

    xs = torch.linspace(low_bound, high_bound, steps=steps)
    ys = torch.ones_like(xs) * center_mu
    total_iterations = len(xs) * len(ys)
    X, Y = np.meshgrid(xs, ys)

    if not load:
        zs = function(Y, X)
        np.savez(output_file_name, zs)
    else:
        zs = np.load(f"{output_dir}/lambda_loss_landscape.npz")["arr_0"]

    plt.plot(xs, zs)
    plt.xlabel("$\lambda$")
    plt.ylabel("Loss")
    plt.show()


if __name__ == '__main__':
    steps = 10
    # center_mu = 384615
    # center_lambda = 576923
    center_mu = 1e4
    center_lambda = 1e4
    loss = get_loss(center_mu, center_lambda)
    print(f"Loss at known optimum: {loss}")
    output_filename = f"{output_dir}/weights.npz"
    # output_filename = f"{output_dir}/lambda_loss_landscape.npz"
    # plot_joined_loss_landscape(output_filename, load=False, steps=steps, center_mu=center_mu,
    #                            center_lambda=center_lambda)
    # plot_joined_loss_landscape(output_filename, load=False, steps=steps, center_mu=center_mu,
    #                            center_lambda=center_lambda, function=vectorized_E, z_title="E")
    plot_joined_loss_landscape(output_filename, steps=steps, center_mu=center_mu, center_lambda=center_lambda, function=vectorized_nu, z_title="$\\nu$")
    # plot_lambda_loss_landscape(center_mu, center_lambda, output_file_name=output_filename, steps=5)
