import argparse
import json
import math
import os

import numpy as np
import torch

import wandb
from examples.training_utils import *
from gradsim import dflex as df
from utils import *

from scipy.spatial import KDTree
from losses import *

from pytorch3d.loss import chamfer_distance

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

    # TODO: CLEAN UP
    # if "training_mesh" not in training_config:
    #     points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    # else:
    #     print(f"Using Training mesh {training_config['training_mesh']}")
    #     points, tet_indices = load_mesh(training_config["training_mesh"])
    points, tet_indices = load_tet_directory("/media/pavlos/One Touch/datasets/gt_generation/magic-salad/tetrahedrals")
    tet_count = len(tet_indices) // 4
    print(f"Fitting simulation with {len(points)} particles and {tet_count} tetrahedra")

    r2 = eval(training_config["sim_mesh_rotation"])
    sim_scale = training_config["sim_scale"]

    positions_pseudo_gt = load_gt_positions(training_config)

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
    gt_E, gt_nu = get_ground_truth_young(simulation_config)
    gt_E, gt_nu = torch.tensor(gt_E), torch.tensor(gt_nu)

    # Initialize models
    position = tuple((0, 0, 0))  # particles are already aligned with GT
    # velocity = tuple(simulation_config["initial_velocity"])
    velocity = tuple((-1, -1, 0))
    scale = 1.0
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    model = model_factory(position, df.quat_identity(), scale, velocity, points, tet_indices, density, k_mu,
                          k_lambda, k_damp)

    initial_velocity_estimate = sim_scale * torch.mean((positions_pseudo_gt[6] - positions_pseudo_gt[0]), dim=0) / 6
    # initial_velocity_estimate = torch.tensor([-1, -1, 0.0])
    gt_mass = model.particle_inv_mass.clone()
    # initial_mu = torch.tensor(1e4) + 100 * torch.rand(1)

    if "mu_initialization" not in training_config:
        initial_mu = torch.rand(1) * 1e3
    else:
        initial_mu = eval(training_config["mu_initialization"])

    if "lambda_initialization" not in training_config:
        initial_lambda = torch.rand(1) * 1e3
    else:
        initial_lambda = eval(training_config["lambda_initialization"])
    # initial_masses = model.particle_inv_mass + 10*torch.rand_like(model.particle_inv_mass)
    initial_masses = 50 * torch.rand_like(model.particle_inv_mass)
    physical_model = PhysicalModel(initial_mu=initial_mu,
                                   initial_lambda=initial_lambda,
                                   initial_velocity=initial_velocity_estimate,
                                   initial_masses=initial_masses,
                                   update_scale_velocity=1.0, update_scale_lame=1, update_scale_masses=1.0)

    if "checkpoint_path" in training_config:
        checkpoint_path = training_config["checkpoint_path"]
        physical_model.load_state_dict(torch.load(f"{checkpoint_path}/physical_model.pth"))

    optimizer = initialize_optimizer(training_config, physical_model)
    # fixed_corr_loss = FixedCorrespondenceDistanceLoss(positions_pseudo_gt[0], points)
    # closest_loss = ClosestOnlyLoss(positions_pseudo_gt[0], points)

    # Do one run before training to get full duration unoptimized
    with torch.no_grad():
        unoptimized_positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                                      scale, velocity, points, tet_indices, density,
                                                      k_mu, k_lambda, k_damp, eval_sim_steps,
                                                      sim_dt, render_steps, physical_model)

    save_positions(unoptimized_positions, f"{output_directory}/unoptimized.npz")

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

        if "loss_start_frame" in training_config and "loss_end_frame" in training_config:
            start = training_config["loss_start_frame"]
            end = training_config["loss_end_frame"]
            # TODO THIS IS MESSED UP NOW, FIX IT
            # loss = training_frame_count * mse(positions[start:end], positions_pseudo_gt[start:end])
            loss = chamfer_distance(positions, positions_pseudo_gt)[0]
        else:
            # weights = torch.zeros(training_frame_count)
            # C = 1
            # weights[12:16] = 6
            # loss = (weights * torch.sum(torch.mean(C * (positions - positions_pseudo_gt) ** 2, dim=1), dim=1)).sum()
            # loss = mse(positions, positions_pseudo_gt)
            # loss = chamfer_distance(positions, positions_pseudo_gt)[0]
            loss = len(positions) * chamfer_distance(positions[17:20], positions_pseudo_gt[17:20])[0]

            # loss = chamfer_distance(positions, positions_pseudo_gt)[0]
        loss.backward()
        optimizer.step()

        if (e % training_config["logging_interval"] == 0 or e == epochs - 1) or loss < 2e-3:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")
            losses.append(loss.item())

            estimated_mu = torch.mean(model.tet_materials[:, 0])
            estimated_lambda = torch.mean(model.tet_materials[:, 1])

            mu_estimates.append(estimated_mu.item())
            lambda_esimates.append(estimated_lambda.item())

            print(f"Mu estimate: {estimated_mu}")
            print(f"Lambda estimate: {estimated_lambda}")

            mu_loss = torch.log10(estimated_mu) - torch.log10(gt_mu)
            mu_mape = torch.abs((gt_mu - estimated_mu) / gt_mu)
            lambda_loss = torch.log10(estimated_lambda) - torch.log10(gt_lambda)
            lambda_mape = torch.abs((gt_lambda - estimated_lambda) / gt_lambda)

            estimated_E, estimated_nu = young_from_lame(estimated_mu, estimated_lambda)
            print(f"E estimate: {estimated_E}")
            print(f"Nu estimate: {estimated_nu}")
            e_log_error = torch.abs(torch.log10(estimated_E) - torch.log10(gt_E))
            nu_error = torch.abs(estimated_nu - gt_nu)

            velocity_estimate_difference = torch.linalg.norm(initial_velocity_estimate - average_initial_velocity)
            print(f"Velocity estimate: {average_initial_velocity}")

            predicted_masses = model.particle_inv_mass
            mass_sqrd_error = torch.nn.functional.mse_loss(predicted_masses, gt_mass)

            wandb.log({"Loss": loss.item(),
                       "Mu": estimated_mu,
                       "Mu Relative Error": mu_mape,
                       "Mu Grad": physical_model.mu_update.grad,
                       "Lambda": estimated_lambda,
                       "Lambda Relative Error": lambda_mape,
                       "Lambda Grad": physical_model.lambda_update.grad,
                       "Velocity Estimate Difference": velocity_estimate_difference,
                       # "Velocity Grad magnitude": torch.linalg.norm(physical_model.velocity_update.grad).item(),
                       "LogE Abs Error": e_log_error,
                       "Nu Abs Error": nu_error})

        if loss < 0.0002:
            # We converged.
            break

        optimizer.zero_grad()

    # Make and save a numpy array of the states (for ease of loading into Blender)
    save_positions(positions, f"{output_directory}/predicted.npz")

    # Save models
    torch.save(physical_model.state_dict(), f"{output_directory}/physical_model.pth")

    # Evaluate
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(), scale, velocity, points,
                                          tet_indices, density, k_mu, k_lambda, k_damp, eval_sim_steps, sim_dt,
                                          render_steps, physical_model)
    save_positions(positions, f"{output_directory}/eval.npz")

    # Log loss landscapes
    wandb_log_curve(mu_estimates, losses, "mu", "Loss", "Mu Loss Landscape", id="mu_losses")
    wandb_log_curve(lambda_esimates, losses, "lambda", "Loss", "Lambda Loss Landscape", id="lambda_losses")
