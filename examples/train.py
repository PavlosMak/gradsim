import argparse
import json
import math
import os

import numpy as np
import torch

import wandb
from examples.training_utils import model_factory, SimpleModel, forward_pass

from gradsim import dflex as df
from utils import load_mesh, lossfn, get_ground_truth_lame, load_pseudo_gt_mesh

from scipy.spatial import KDTree

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

    r = df.quat_from_axis_angle((1.0, 0.0, 0.0), 1.0 * math.pi)
    r2 = df.quat_from_axis_angle((1.0, 0.0, 0.0), -0.5 * math.pi)
    sim_scale = training_config["sim_scale"]

    path_to_gt = training_config["path_to_gt"]
    positions_pseudo_gt = np.load(path_to_gt)["arr_0"]
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

    mass_model = SimpleModel(
        model.particle_inv_mass + 10 * torch.rand_like(model.particle_inv_mass),
        activation=torch.nn.functional.relu
    )

    initial_velocity_estimate = sim_scale * torch.mean(positions_pseudo_gt[1] - positions_pseudo_gt[0], dim=0)
    velocity_model = SimpleModel(initial_velocity_estimate)
    k_mu_model = SimpleModel(1e5 * torch.rand(1), activation=torch.nn.functional.relu)
    k_lambda_model = SimpleModel(1e5 * torch.rand(1), activation=torch.nn.functional.relu)

    if "checkpoint_path" in training_config:
        checkpoint_path = training_config["checkpoint_path"]
        mass_model.load_state_dict(torch.load(f"{checkpoint_path}/mass_model.pth"))
        velocity_model.load_state_dict(torch.load(f"{checkpoint_path}/velocity_model.pth"))
        k_mu_model.load_state_dict(torch.load(f"{checkpoint_path}/mu_model.pth"))
        k_lambda_model.load_state_dict(torch.load(f"{checkpoint_path}/lambda_model.pth"))

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
                                                      sim_dt, render_steps)
        positions_np = np.array([p.detach().cpu().numpy() for p in unoptimized_positions])
        np.savez(os.path.join(output_directory, "unoptimized.npz"), positions_np)

    epochs = training_config["epochs"]
    for e in range(epochs):
        positions, model, state, average_initial_velocity = forward_pass(velocity_model, k_mu_model, k_lambda_model,
                                                                         mass_model, position, df.quat_identity(),
                                                                         scale, velocity, points, tet_indices, density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         training_sim_steps, sim_dt, render_steps)

        loss = lossfn(positions, positions_pseudo_gt, index_map)

        if e % training_config["logging_interval"] == 0 or e == epochs - 1:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")

            estimated_mu = torch.mean(model.tet_materials[:, 0])
            estimated_lambda = torch.mean(model.tet_materials[:, 1])

            mu_loss = torch.log10(torch.abs(estimated_mu - gt_mu))
            lambda_loss = torch.log10(torch.abs(estimated_lambda - gt_lambda))

            wandb.log({"Loss": loss.item(),
                       "Mu": estimated_mu,
                       "Mu Abs Error Log10": mu_loss,
                       "Lambda": estimated_lambda,
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
                                          tet_indices, density, k_mu, k_lambda, k_damp, eval_sim_steps, sim_dt,
                                          render_steps)
        positions_np = np.array([p.detach().cpu().numpy() for p in positions])
        np.savez(f"{output_directory}/eval.npz", positions_np)
