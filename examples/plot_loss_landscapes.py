import matplotlib.pyplot as plt
import numpy as np
import torch
import json

from examples.utils import load_mesh
from gradsim import dflex as df

from examples.training_utils import load_gt_positions, forward_pass

path_to_config = "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/sampledata/configs/thinner_torus.json"

with open(path_to_config) as config_file:
    simulation_config = json.load(config_file)

training_config = simulation_config["training"]

positions_pseudo_gt = load_gt_positions(training_config)

position = tuple((0, 0, 0))
velocity = tuple(simulation_config["initial_velocity"])
scale = simulation_config["scale"]
density = simulation_config["density"]
k_mu = simulation_config["mu"]
k_lambda = simulation_config["lambda"]
k_damp = simulation_config["damp"]

points, tet_indices = load_mesh(training_config["training_mesh"])

training_frame_count = training_config["frame_count"]
sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
training_sim_steps = int(training_sim_duration / sim_dt)

render_steps = simulation_config["sim_substeps"]

mse = torch.nn.MSELoss(reduction="sum")

total = 0
i = 0


def get_loss(x, y) -> float:
    global i
    print(f"{i}/{total}")
    i += 1
    k_mu = x
    k_lambda = y
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                          scale, velocity, points, tet_indices, density,
                                          k_mu, k_lambda, k_damp, training_sim_steps,
                                          sim_dt, render_steps)

        loss = mse(positions, positions_pseudo_gt[:training_frame_count])
    return loss.item()


vectorized_fun = np.vectorize(get_loss)


def plot_mu_loss_landscape():
    xs = torch.linspace(5e3, 5e4, steps=100)
    ys = torch.linspace(5e3, 5e4, steps=100)
    global total
    total = len(xs) * len(ys)
    X, Y = np.meshgrid(xs, ys)
    Z = vectorized_fun(X, Y)
    np.savez("/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/output/joined_loss_landscape.npz", Z)
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('$\mu$')
    ax.set_ylabel('$\lambda$')
    ax.set_zlabel('Loss')
    plt.show()


def plot_lambda_loss_landscape():
    pass


def plot_joined_loss_landscape():
    pass


if __name__ == '__main__':


    plot_mu_loss_landscape()
