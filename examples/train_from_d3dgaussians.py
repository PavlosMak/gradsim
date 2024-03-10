import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

import wandb

from gradsim import dflex as df
from utils import export_obj, load_mesh, lossfn, get_ground_truth_lame

from scipy.spatial import KDTree

from pytorch3d.loss import chamfer_distance


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param, activation=None, update_scale=0.1):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape) * update_scale)
        self.param = param
        self.activation = activation

    def forward(self):
        out = self.param + self.update
        if self.activation is not None:
            return self.activation(out) + 1e-8
        return out


def model_factory(pos, rot, scale, vel, vertices, tet_indices, density, k_mu, k_lambda, k_damp):
    builder = df.sim.ModelBuilder()
    builder.add_soft_mesh(pos=pos, rot=rot, scale=scale, vel=vel,
                          vertices=vertices, indices=tet_indices, density=density,
                          k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
    model = builder.finalize("cpu")

    model.tri_ke = 0.0
    model.tri_ka = 0.0
    model.tri_kd = 0.0
    model.tri_kb = 0.0

    model.contact_ke = 1.0e4
    model.contact_kd = 1.0
    model.contact_kf = 10.0
    model.contact_mu = 0.5

    model.particle_radius = 0.05
    model.ground = True

    return model


def forward_pass(velocity_model, k_mu_model, k_lambda_model, mass_model, position, r, scale, velocity,
                 points,
                 tet_indices, density, k_mu, k_lambda, k_damp, sim_steps, sim_dt):
    model = model_factory(position, r, scale, velocity, points, tet_indices, density, k_mu, k_lambda, k_damp)
    # Infer model parameters
    model.particle_v[:, ] = velocity_model()
    model.tet_materials[:, 0] = k_mu_model()
    model.tet_materials[:, 1] = k_lambda_model()
    model.particle_inv_mass = mass_model()

    average_initial_velocity = torch.mean(model.particle_v, dim=0)

    integrator = df.sim.SemiImplicitIntegrator()
    sim_time = 0.0
    state = model.state()

    faces = model.tri_indices
    export_obj(state.q.clone().detach().cpu().numpy(), faces,
               os.path.join(output_directory, "simulation_mesh.obj"))

    positions = []

    for i in trange(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)

        sim_time += sim_dt

        if i % render_steps == 0:
            positions.append(state.q)

    positions = torch.stack(positions)

    return positions, model, state, average_initial_velocity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory to store experiment logs in.")
    parser.add_argument("--sim-config", type=str, default=os.path.join("sampledata", "configs", "spot.json"),
                        help="Path to simulation configuration variables")
    args = parser.parse_args()
    device = "cuda:0"
    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)

    training_config = simulation_config["training"]
    torch.manual_seed(training_config["seed"])

    run = wandb.init(project="Gaussian Inverse Physics", config=training_config)
    output_directory = f"{training_config['logdir']}/{run.name}"
    os.makedirs(output_directory)
    # torch.autograd.set_detect_anomaly(True)  # Uncomment to debug backpropagation

    training_frame_count = training_config["frame_count"]
    # sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
    sim_dt = 1 / 9600
    training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
    training_sim_steps = int(training_sim_duration / sim_dt)

    render_steps = simulation_config["sim_substeps"]
    path_to_exp = f"{training_config['path_to_gt']}/{training_config['exp_name']}"
    path_to_gt = f"{path_to_exp}/gt"

    # points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    points, tet_indices = load_mesh("/media/pavlos/One Touch/datasets/gt_generation/fearless-microwave/torus.obj")
    tet_count = len(tet_indices) // 4
    print(f"Fitting simulation with {len(points)} particles and {tet_count} tetrahedra")

    # r = df.quat_multiply(
    #     df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 1.0),
    #     df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.0),  # we need to reflect around the x-axis
    # )

    # r = df.quat_identity()
    r = df.quat_from_axis_angle((1.0, 0.0, 0.0), 1.0 * math.pi)
    r2 = df.quat_from_axis_angle((1.0, 0.0, 0.0), -0.5 * math.pi)
    sim_scale = training_config["sim_scale"]

    path_to_pseudo_gt = "/media/pavlos/One Touch/datasets/gt_generation/fearless-microwave/pseudo_gt_positions.npz"
    positions_pseudo_gt = np.load(path_to_pseudo_gt)["arr_0"]
    edited_positions = []
    training_frame_count = positions_pseudo_gt.shape[0]
    for i in range(training_frame_count):
        frame_positions = torch.tensor(positions_pseudo_gt[i], dtype=torch.float32)
        rotation_matrix = torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
        frame_positions = sim_scale * (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
        edited_positions.append(frame_positions)
    positions_pseudo_gt = torch.stack(edited_positions)
    floor_level_offset = torch.zeros(3)
    floor_level_offset[1] = torch.min(positions_pseudo_gt[:, :, 1].flatten())
    positions_pseudo_gt -= floor_level_offset

    # Load real ground truth for 'eval' and also save it in the output folder of this script
    gt_positions = \
        np.load("/media/pavlos/One Touch/datasets/inverse_physics_results/scarlett-terrain-117/positions_gt.npz")[
            "arr_0"]
    np.savez(f"{training_config['logdir']}/gt_positions.npz", gt_positions)
    gt_positions = torch.tensor(gt_positions)

    full_frame_count = gt_positions.shape[0]
    eval_sim_duration = full_frame_count / simulation_config["physics_engine_rate"]
    eval_sim_steps = int(eval_sim_duration / sim_dt)

    gt_positions = gt_positions[:training_frame_count]

    gt_chamfer_distance = chamfer_distance(positions_pseudo_gt, gt_positions, batch_reduction="mean")[0]
    print(f"Pseudo GT Chamfer Distance: {gt_chamfer_distance.item()}")

    # Correct point coordinate system
    points = sim_scale * (df.quat_to_matrix(r2) @ points.transpose(1, 0)).transpose(1, 0)

    # Log ground truth positions
    positions_np = np.array([p.detach().cpu().numpy() for p in positions_pseudo_gt])
    np.savez(f"{output_directory}/pseudo_gt_positions.npz", positions_np)

    gt_mu, gt_lambda = get_ground_truth_lame(simulation_config)
    gt_mu, gt_lambda = torch.tensor(gt_mu), torch.tensor(gt_lambda)

    # Create correspondences
    tree = KDTree(positions_pseudo_gt[0])
    index_map = {}
    for pi, p in enumerate(points):
        dist, index = tree.query(p)
        index_map[pi] = (index, dist)

    # Initialize models
    position = tuple((0, 0, 0))  # particles are already aligned with GT
    velocity = tuple(simulation_config["initial_velocity"])
    scale = 1.0
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    model = model_factory(position, df.quat_identity(), scale, velocity, points, tet_indices, density, k_mu,
                          k_lambda, k_damp)

    mass_model = SimpleModel(
        model.particle_inv_mass + 10 * torch.rand_like(model.particle_inv_mass),
        activation=torch.nn.functional.relu
    )

    initial_velocity_estimate = torch.mean(positions_pseudo_gt[1] - positions_pseudo_gt[0], dim=0)
    velocity_model = SimpleModel(initial_velocity_estimate)
    k_mu_model = SimpleModel(
        1e5 * torch.rand(1),
        activation=torch.nn.functional.relu
    )
    k_lambda_model = SimpleModel(
        1e5 * torch.rand(1),
        activation=torch.nn.functional.relu
    )

    parameters = [
        list(k_mu_model.parameters())[0],
        list(k_lambda_model.parameters())[0],
        list(mass_model.parameters())[0],
        list(velocity_model.parameters())[0],
    ]

    optimizer = torch.optim.Adam(parameters, lr=training_config["lr"])

    # Do one run before training to get full duration unoptimized
    with torch.no_grad():
        unoptimized_positions, _, _, _ = forward_pass(velocity_model, k_mu_model, k_lambda_model,
                                                      mass_model, position, df.quat_identity(),
                                                      scale, velocity, points, tet_indices, density,
                                                      k_mu, k_lambda, k_damp, eval_sim_steps,
                                                      sim_dt)
        positions_np = np.array([p.detach().cpu().numpy() for p in unoptimized_positions])
        np.savez(os.path.join(output_directory, "unoptimized.npz"), positions_np)

    epochs = training_config["epochs"]
    for e in range(epochs):
        positions, model, state, average_initial_velocity = forward_pass(velocity_model, k_mu_model, k_lambda_model,
                                                                         mass_model, position, df.quat_identity(),
                                                                         scale, velocity, points, tet_indices, density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         training_sim_steps, sim_dt)

        loss = lossfn(positions, positions_pseudo_gt, index_map)

        if e % training_config["logging_interval"] == 0 or e == epochs - 1:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")

            average_mu = torch.mean(model.tet_materials[:, 0])
            average_lambda = torch.mean(model.tet_materials[:, 1])

            mu_loss = torch.abs(torch.log10(average_mu) - torch.log10(gt_mu))
            lambda_loss = torch.abs(torch.log10(average_lambda) - torch.log10(gt_lambda))

            wandb.log({"Loss": loss.item(),
                       "Pseudo GT Chamfer Distace": gt_chamfer_distance.item(),
                       "Mu": average_mu,
                       "Mu Abs Error Log10": mu_loss,
                       "Lambda": average_lambda,
                       "Lambda Abs Error Log10": lambda_loss})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Make and save a numpy array of the states (for ease of loading into Blender)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(os.path.join(output_directory, f"predicted.npz"), positions_np)

    # Save models
    torch.save(mass_model.state_dict(), f"{output_directory}/mass_model.pth")
    torch.save(velocity_model.state_dict(), f"{output_directory}/velocity_model.pth")
    torch.save(k_mu_model.state_dict(), f"{output_directory}/mu_model.pth")
    torch.save(k_lambda_model.state_dict(), f"{output_directory}/lambda_model.pth")

    # Evaluate
    with torch.no_grad():
        positions, _, _, _ = forward_pass(velocity_model, k_mu_model, k_lambda_model, mass_model,
                                          position, df.quat_identity(), scale, velocity, points,
                                          tet_indices, density, k_mu, k_lambda, k_damp, eval_sim_steps, sim_dt)
        positions_np = np.array([p.detach().cpu().numpy() for p in positions])
        np.savez(f"{output_directory}/eval.npz", positions_np)
