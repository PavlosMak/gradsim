import argparse
import json
import math
import os

import numpy as np
import torch

import wandb
from examples.training_utils import model_factory, forward_pass, initialize_optimizer, PhysicalModel

from gradsim import dflex as df
from utils import load_mesh, lossfn, get_ground_truth_lame, load_pseudo_gt_mesh

from scipy.spatial import KDTree

from logging_utils import wandb_log_curve

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
    sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
    training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
    training_sim_steps = int(training_sim_duration / sim_dt)

    render_steps = simulation_config["sim_substeps"]
    path_to_exp = f"{training_config['path_to_gt']}/{training_config['exp_name']}"

    if "training_mesh" not in training_config:
        points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    else:
        print(f"Using Training mesh {training_config['training_mesh']}")
        points, tet_indices = load_mesh(training_config["training_mesh"])
    tet_count = len(tet_indices) // 4
    print(f"Fitting simulation with {len(points)} particles and {tet_count} tetrahedra")

    r = eval(training_config["gt_rotation"])
    r2 = eval(training_config["sim_mesh_rotation"])
    sim_scale = training_config["sim_scale"]

    path_to_gt = training_config["path_to_gt"]
    positions_pseudo_gt = np.load(path_to_gt)["arr_0"]
    edited_positions = []
    if "transform_gt_points" in training_config:
        for i in range(training_frame_count):
            frame_positions = torch.tensor(positions_pseudo_gt[i], dtype=torch.float32)
            rotation_matrix = torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
            # frame_positions = sim_scale * (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
            # TODO: MAKE CONTROL OF GT SCALE SEPARATE
            frame_positions = (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
            edited_positions.append(frame_positions)
        positions_pseudo_gt = torch.stack(edited_positions)
        # TODO: ALSO MAKE THIS NICER
        # floor_level_offset = torch.zeros(3)
        # floor_level_offset[1] = torch.min(positions_pseudo_gt[:, :, 1].flatten())
        # positions_pseudo_gt -= floor_level_offset
    else:
        positions_pseudo_gt = torch.tensor(positions_pseudo_gt)

    full_frame_count = training_config["eval_for"]
    eval_sim_duration = full_frame_count / simulation_config["physics_engine_rate"]
    eval_sim_steps = int(eval_sim_duration / sim_dt)

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

    initial_velocity_estimate = sim_scale * torch.mean(positions_pseudo_gt[1] - positions_pseudo_gt[0], dim=0)
    physical_model = PhysicalModel(initial_mu=torch.tensor(1e4),
                                   initial_lambda=torch.tensor(1e4),
                                   initial_velocity=torch.tensor([velocity[0], velocity[1], velocity[2]],
                                                                 dtype=torch.float32),
                                   initial_masses=model.particle_inv_mass + 0.1 * torch.rand_like(
                                       model.particle_inv_mass),
                                   update_scale_velocity=0.0, update_scale_lame=10, update_scale_masses=0)

    if "checkpoint_path" in training_config:
        checkpoint_path = training_config["checkpoint_path"]
        physical_model.load_state_dict(torch.load(f"{checkpoint_path}/physical_model.pth"))

    optimizer = initialize_optimizer(training_config, physical_model)

    # Do one run before training to get full duration unoptimized
    with torch.no_grad():
        unoptimized_positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                                      scale, velocity, points, tet_indices, density,
                                                      k_mu, k_lambda, k_damp, eval_sim_steps,
                                                      sim_dt, render_steps, physical_model)
    positions_np = np.array([p.detach().cpu().numpy() for p in unoptimized_positions])
    np.savez(os.path.join(output_directory, "unoptimized.npz"), positions_np)

    epochs = training_config["epochs"]
    losses = []
    mu_estimates = []
    lambda_esimates = []

    mse = torch.nn.MSELoss(reduction="sum")

    for e in range(epochs):
        positions, model, state, average_initial_velocity = forward_pass(position, df.quat_identity(),
                                                                         scale, velocity, points, tet_indices, density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         training_sim_steps, sim_dt, render_steps,
                                                                         physical_model)

        # loss = lossfn(positions, positions_pseudo_gt, index_map)
        loss = mse(positions, positions_pseudo_gt[:training_frame_count])
        loss.backward()
        optimizer.step()

        if e % training_config["logging_interval"] == 0 or e == epochs - 1:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")
            losses.append(loss.item())

            estimated_mu = torch.mean(model.tet_materials[:, 0])
            estimated_lambda = torch.mean(model.tet_materials[:, 1])

            mu_estimates.append(estimated_mu.item())
            lambda_esimates.append(estimated_lambda.item())

            print(f"Mu estimate: {estimated_mu}")
            print(f"Lambda estimate: {estimated_lambda}")

            mu_loss = torch.log10(torch.abs(estimated_mu - gt_mu))
            lambda_loss = torch.log10(torch.abs(estimated_lambda - gt_lambda))

            velocity_estimate_difference = torch.linalg.norm(initial_velocity_estimate - average_initial_velocity)
            print(f"Velocity estimate: {average_initial_velocity}")

            wandb.log({"Loss": loss.item(),
                       "Mu": estimated_mu,
                       "Mu Abs Error Log10": mu_loss,
                       "Lambda": estimated_lambda,
                       "Lambda Abs Error Log10": lambda_loss,
                       "Velocity Estimate Difference": velocity_estimate_difference})

        optimizer.zero_grad()

    # Make and save a numpy array of the states (for ease of loading into Blender)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(os.path.join(output_directory, f"predicted.npz"), positions_np)

    # Save models
    torch.save(physical_model.state_dict(), f"{output_directory}/physical_model.pth")

    # Evaluate
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(), scale, velocity, points,
                                          tet_indices, density, k_mu, k_lambda, k_damp, eval_sim_steps, sim_dt,
                                          render_steps, physical_model)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(f"{output_directory}/eval.npz", positions_np)

    # Log loss landscapes
    wandb_log_curve(mu_estimates, losses, "mu", "Loss", "Mu Loss Landscape", id="mu_losses")
    wandb_log_curve(lambda_esimates, losses, "lambda", "Loss", "Lambda Loss Landscape", id="lambda_losses")
