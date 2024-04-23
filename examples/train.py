import argparse
import json
import math  # do not remove - required for parsing the configs
import os

import numpy as np
import torch

import wandb
from examples.training_utils import *
from gradsim import dflex as df
from utils import *

from losses import *

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

    training_sim_steps = frames_to_sim_steps(training_frame_count, simulation_config["physics_engine_rate"], sim_dt)

    render_steps = simulation_config["sim_substeps"]
    path_to_exp = f"{training_config['path_to_gt']}/{training_config['exp_name']}"

    if "training_mesh" not in training_config:
        points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    else:
        print(f"Using Training mesh {training_config['training_mesh']}")
        points, tet_indices, _ = load_mesh(training_config["training_mesh"])
    tet_count = len(tet_indices) // 4
    print(f"Fitting simulation with {len(points)} particles and {tet_count} tetrahedra")

    np.savez(f"{output_directory}/vertices.npz", points)
    np.savez(f"{output_directory}/tets.npz", tet_indices)

    r2 = eval(training_config["sim_mesh_rotation"])
    sim_scale = training_config["sim_scale"]

    positions_pseudo_gt, floor_level_offset = load_gt_positions(training_config)

    eval_sim_steps = frames_to_sim_steps(training_config["eval_for"], simulation_config["physics_engine_rate"], sim_dt)

    # Correct point coordinate system
    points = sim_scale * (df.quat_to_matrix(r2) @ points.transpose(1, 0)).transpose(1, 0)
    points = points + floor_level_offset.numpy()

    # Log ground truth positions
    positions_np = np.array([p.detach().cpu().numpy() for p in positions_pseudo_gt])
    np.savez(f"{output_directory}/pseudo_gt_positions.npz", positions_np)

    gt_mu, gt_lambda = get_ground_truth_lame(simulation_config)
    gt_mu, gt_lambda = torch.tensor(gt_mu), torch.tensor(gt_lambda)
    gt_E, gt_nu = get_ground_truth_young(simulation_config)
    gt_E, gt_nu = torch.tensor(gt_E), torch.tensor(gt_nu)

    print(f"Ground truth: \n E: {gt_E}; nu: {gt_nu}")

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

    gt_mass = model.particle_inv_mass.clone()

    initial_E = initialize_from_config(training_config, "E_initialization", torch.tensor(1e3))
    initial_nu = initialize_from_config(training_config, "nu_initialization", torch.tensor(0.1))
    initial_mu, initial_lambda = lame_from_young(initial_E, initial_nu)

    print(f"Initial values: \n E: {initial_E}; nu: {initial_nu}")

    optimization_set = set(training_config["optimize"])

    initial_masses = 50 * torch.rand_like(model.particle_inv_mass)
    # physical_model = PhysicalModel(initial_mu=initial_mu,
    #                                initial_lambda=initial_lambda,
    #                                initial_velocity=initial_velocity_estimate,
    #                                initial_masses=initial_masses,
    #                                update_scale_velocity=1.0, update_scale_lame=1, update_scale_masses=1.0)

    physical_model = PhysicalModelYoungPoisson(initial_E=initial_E, initial_nu=initial_nu,
                                               initial_masses=initial_masses, update_scale_masses=1.0)

    if "checkpoint_path" in training_config:
        checkpoint_path = training_config["checkpoint_path"]
        print(f"Loading checkpoint: {checkpoint_path}")
        physical_model.load_state_dict(torch.load(checkpoint_path))
        print(
            f"Loaded parameter values:\n E: {10 ** physical_model.global_E};  nu: {constraint(physical_model.global_nu, [-0.45, 0.45])} ")

    fix_top_plane = False
    if "fix_top_plane" in simulation_config:
        fix_top_plane = simulation_config["fix_top_plane"]

    if "gt_noise_scale" in training_config:
        scale = training_config["gt_noise_scale"]
        positions_pseudo_gt = positions_pseudo_gt + scale * torch.rand_like(positions_pseudo_gt)

    # Do one run before training to get full duration unoptimized
    with torch.no_grad():
        unoptimized_positions, _, _, _ = forward_pass(position, df.quat_identity(),
                                                      scale, velocity, points, tet_indices, density,
                                                      k_mu, k_lambda, k_damp, eval_sim_steps,
                                                      sim_dt, render_steps, physical_model, fix_top_plane=fix_top_plane,
                                                      optimization_set=optimization_set)

    save_positions(unoptimized_positions, f"{output_directory}/unoptimized.npz")

    epochs = training_config["epochs"]
    losses = []
    mu_estimates = []
    lambda_esimates = []

    lossfn = loss_factory(training_config)

    velocity_sim_steps = frames_to_sim_steps(training_config["velocity_frames"],
                                             simulation_config["physics_engine_rate"], sim_dt)
    velocity_epochs = training_config["velocity_epochs"]
    velocity_optimizer = initialize_velocity_optimizer(training_config, physical_model)

    # def closure():
    #     velocity_optimizer.zero_grad()
    #     positions, model, state, average_initial_velocity = forward_pass(position, df.quat_identity(),
    #                                                                      scale, velocity, points, tet_indices,
    #                                                                      density,
    #                                                                      k_mu, k_lambda, k_damp,
    #                                                                      training_sim_steps, sim_dt, render_steps,
    #                                                                      physical_model,
    #                                                                      fix_top_plane=fix_top_plane,
    #                                                                      optimization_set=optimization_set)
    #     loss = lossfn(positions[:3], positions_pseudo_gt[:3])
    #     loss.backward()
    #     print(f"Loss: {loss.item()}")
    #     wandb.log({"Loss": loss.item()})
    #     return loss

    for e in range(velocity_epochs):
        if "velocity" not in optimization_set:
            break

        velocity_optimizer.zero_grad()
        positions, model, state, average_initial_velocity = forward_pass(position, df.quat_identity(),
                                                                         scale, velocity, points, tet_indices,
                                                                         density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         velocity_sim_steps, sim_dt, render_steps,
                                                                         physical_model,
                                                                         fix_top_plane=fix_top_plane,
                                                                         optimization_set=optimization_set)
        loss = lossfn(positions, positions_pseudo_gt[:training_config["velocity_frames"]])
        loss.backward()
        velocity_optimizer.step()

        # velocity_optimizer.step(closure=closure)
        #
        if (e % training_config["logging_interval"] == 0 or e == velocity_epochs - 1):
            print(f"Velocity Epoch: {(e + 1):03d}/{velocity_epochs:03d}")
            print(f"Velocity estimate: {physical_model.global_velocity.data}")
            print(f"Loss: {loss.item()}")
            wandb.log({"Velocity Estimate Difference": torch.linalg.norm(physical_model.global_velocity),
                       "Velocity Grad magnitude": torch.linalg.norm(physical_model.global_velocity.grad).item(),
                       "Mean Mass update": torch.mean(physical_model.mass_updates).item(),
                       "Loss": loss.item()
                       })

    torch.save(physical_model.state_dict(), f"{output_directory}/physical_model_optimized_velocity.pth")

    # optimizer = initialize_optimizer(training_config, physical_model)
    optimizer = initialize_optimizer_young_poisson(training_config, physical_model)
    for e in range(epochs):
        positions, model, state, average_initial_velocity = forward_pass(position, df.quat_identity(),
                                                                         scale, velocity, points, tet_indices, density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         training_sim_steps, sim_dt, render_steps,
                                                                         physical_model, fix_top_plane=fix_top_plane,
                                                                         optimization_set=optimization_set)

        loss = lossfn(positions, positions_pseudo_gt)
        loss.backward()
        optimizer.step()

        if (e % training_config["logging_interval"] == 0 or e == epochs - 1):
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

            velocity_estimate_difference = torch.linalg.norm(average_initial_velocity)
            print(f"Velocity estimate: {average_initial_velocity}")

            predicted_masses = model.particle_inv_mass
            mass_sqrd_error = torch.nn.functional.mse_loss(predicted_masses, gt_mass)

            wandb.log({"Loss": loss.item(),
                       "Mu": estimated_mu,
                       "Mu Relative Error": mu_mape,
                       "E Grad": physical_model.global_E.grad,
                       # "Mu Grad": physical_model.mu_update.grad,
                       "Lambda": estimated_lambda,
                       "Lambda Relative Error": lambda_mape,
                       "Nu Grad": physical_model.global_nu.grad,
                       # "Lambda Grad": physical_model.lambda_update.grad,
                       "Velocity Estimate Difference": velocity_estimate_difference,
                       # "Velocity Grad magnitude": torch.linalg.norm(physical_model.velocity_update.grad).item(),
                       "LogE Abs Error": e_log_error,
                       "Nu Abs Error": nu_error})

            torch.save(physical_model.state_dict(), f"{output_directory}/physical_model_epoch_{e}.pth")

        optimizer.zero_grad()

    # Make and save a numpy array of the states (for ease of loading into Blender)
    save_positions(positions, f"{output_directory}/predicted.npz")

    # Save models
    torch.save(physical_model.state_dict(), f"{output_directory}/physical_model.pth")

    # Evaluate
    with torch.no_grad():
        positions, _, _, _ = forward_pass(position, df.quat_identity(), scale, velocity, points,
                                          tet_indices, density, k_mu, k_lambda, k_damp, eval_sim_steps, sim_dt,
                                          render_steps, physical_model, fix_top_plane=fix_top_plane,
                                          optimization_set=optimization_set)
    save_positions(positions, f"{output_directory}/eval.npz")

    # Log loss landscapes
    wandb_log_curve(mu_estimates, losses, "mu", "Loss", "Mu Loss Landscape", id="mu_losses")
    wandb_log_curve(lambda_esimates, losses, "lambda", "Loss", "Lambda Loss Landscape", id="lambda_losses")

    run.summary["Final E"] = estimated_E
    run.summary["Final nu"] = estimated_nu
    run.summary["Final Velocity"] = physical_model.global_velocity.data
