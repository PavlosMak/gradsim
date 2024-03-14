import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

import wandb

from gradsim import dflex as df
from utils import load_pseudo_gt_mesh, export_obj, get_tet_volume, get_volumes

from scipy.spatial import KDTree

from pytorch3d.loss import chamfer_distance


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param, activation=None):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape) * 0.1)
        self.param = param
        self.activation = activation

    def forward(self):
        out = self.param + self.update
        if self.activation is not None:
            return self.activation(out) + 1e-8
        return out


def forward_pass(velocity_model, k_mu_model, k_lambda_model, mass_model, position, r, scale, velocity, points,
                 tet_indices, density, k_mu, k_lambda, k_damp, sim_steps, sim_dt):
    builder = df.sim.ModelBuilder()
    builder.add_soft_mesh(pos=position, rot=r, scale=scale, vel=velocity,
                          vertices=points, indices=tet_indices, density=density,
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

    # Infer model parameters
    # model.particle_v = velocity_model()
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
    sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
    training_sim_duration = training_frame_count / simulation_config["physics_engine_rate"]
    training_sim_steps = int(training_sim_duration / sim_dt)

    render_steps = simulation_config["sim_substeps"]
    path_to_exp = f"{training_config['path_to_gt']}/{training_config['exp_name']}"
    path_to_gt = f"{path_to_exp}/gt"

    points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    tet_count = len(tet_indices) // 4
    print(f"Fitting simulation with {len(points)} particles and {tet_count} tetrahedra")

    # r = df.quat_multiply(
    #     df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 1.0),
    #     df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.0),  # we need to reflect around the x-axis
    # )

    # r = df.quat_identity()
    r = df.quat_from_axis_angle((1.0, 0.0, 0.0), 0.5 * math.pi)

    positions_pseudo_gt = []
    pseudo_gt_ground_level = torch.inf
    for progress_counter in range(training_frame_count):
        frame_gt_path = f"{path_to_gt}/gt_{progress_counter}.txt"
        frame_positions = torch.tensor(np.loadtxt(frame_gt_path, delimiter="\t"), dtype=torch.float32)
        rotation_matrix = torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
        frame_positions = (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
        frame_positions = 3 * frame_positions
        frame_min_y = torch.min(frame_positions[:, 1])
        pseudo_gt_ground_level = min(frame_min_y, pseudo_gt_ground_level)
        positions_pseudo_gt.append(frame_positions)

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

    gt_chamfer_distance = chamfer_distance(torch.stack(positions_pseudo_gt), gt_positions, batch_reduction="mean")[0]
    print(f"Pseudo GT Chamfer Distance: {gt_chamfer_distance.item()}")

    # Correct point coordinate system
    points = 3 * (df.quat_to_matrix(r) @ points.transpose(1, 0)).transpose(1, 0)
    # Correct for ground offset
    gt_ground_offset_np = np.array([0, -pseudo_gt_ground_level, 0], dtype=np.float32)
    points += gt_ground_offset_np
    gt_ground_offset = torch.tensor(gt_ground_offset_np)
    positions_pseudo_gt = [p + gt_ground_offset for p in positions_pseudo_gt]

    # Log ground truth positions
    positions_np = np.array([p.detach().cpu().numpy() for p in positions_pseudo_gt])
    np.savez(f"{output_directory}/pseudo_gt_positions.npz", positions_np)

    # For every point in the simulated mesh we need to find the corresponding point in the GT
    tree = KDTree(points)
    common_indices = []
    for p in positions_pseudo_gt[0]:
        _, index = tree.query(p)
        common_indices.append(index)

    # Initialize models
    mass_model = SimpleModel(
        10 * torch.rand(points.shape[0]),
        activation=torch.nn.functional.relu
    )
    velocity_model = SimpleModel(torch.rand(3))
    k_mu_model = SimpleModel(
        # 1e4 * torch.rand(1)
        torch.tensor(9900),
        activation=torch.nn.functional.relu
    )
    k_lambda_model = SimpleModel(
        # 1e4 * torch.rand(1)
        torch.tensor(9900),
        activation=torch.nn.functional.relu
    )

    position = tuple((0, 0, 0))  # particles are already aligned with GT
    velocity = tuple(simulation_config["initial_velocity"])
    scale = 1.0
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    # parameters = [list(mass_model.parameters())[0]]
    parameters = [
        list(k_mu_model.parameters())[0],
        list(k_lambda_model.parameters())[0],
        list(mass_model.parameters())[0],
        list(velocity_model.parameters())[0]
    ]

    optimizer = torch.optim.Adam(parameters, lr=training_config["lr"])
    lossfn = torch.nn.MSELoss()

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
                                                                         k_mu, k_lambda, k_damp, training_sim_steps,
                                                                         sim_dt)

        pred_positions = [p[common_indices] for p in positions]
        loss = sum([lossfn(est, gt) for est, gt in
                    zip(pred_positions[::training_config["compare_every"]],
                        positions_pseudo_gt[::training_config["compare_every"]])])

        if e % training_config["logging_interval"] == 0 or e == epochs - 1:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")
            velocity_loss = torch.linalg.norm(
                average_initial_velocity - torch.tensor(simulation_config["initial_velocity"]))
            print(f"Initial Velocity: {average_initial_velocity} - Initial Velocity Loss: {velocity_loss}")

            predicted_chamfer_distance = chamfer_distance(torch.stack(pred_positions), gt_positions,
                                                          batch_reduction="mean")[0]
            print(f"Predic Chamfer Distance: {predicted_chamfer_distance.item()}\n"
                  f"Pseudo Chamfer Distance: {gt_chamfer_distance.item()}")

            average_mu = torch.mean(model.tet_materials[:, 0])
            average_lambda = torch.mean(model.tet_materials[:, 1])
            mu_error = torch.abs(average_mu - simulation_config["mu"])
            lambda_error = torch.abs(average_lambda - simulation_config["lambda"])
            print(f"Mu error: {mu_error.item()}, Lambda error: {lambda_error.item()}")

            wandb.log({"Loss": loss.item(),
                       "Chamfer Distance": predicted_chamfer_distance.item(),
                       "Initial Velocity Loss": velocity_loss.item(),
                       "Pseudo GT Chamfer Distace": gt_chamfer_distance.item(),
                       "Mu error": mu_error.item(),
                       "Lambda error": lambda_error.item(),
                       "Mu": model.tet_materials[:, 0][0].item(),
                       "Lambda": model.tet_materials[:, 1][0].item()
                       })

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Make and save a numpy array of the states (for ease of loading into Blender)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(os.path.join(output_directory, f"predicted.npz"), positions_np)

    gt_chamfer_distance = chamfer_distance(torch.stack(positions_pseudo_gt), gt_positions, batch_reduction="mean")[0]

    # Save models
    torch.save(mass_model.state_dict(), f"{output_directory}/mass_model.pth")
    torch.save(velocity_model.state_dict(), f"{output_directory}/velocity_model.pth")
    torch.save(k_mu_model.state_dict(), f"{output_directory}/mu_model.pth")
    torch.save(k_lambda_model.state_dict(), f"{output_directory}/lambda_model.pth")

    # Evaluate
    with torch.no_grad():
        positions, _, _, _ = forward_pass(velocity_model, k_mu_model, k_lambda_model, mass_model, position,
                                          df.quat_identity(), scale, velocity, points, tet_indices, density, k_mu,
                                          k_lambda, k_damp, eval_sim_steps, sim_dt)
        positions_np = np.array([p.detach().cpu().numpy() for p in positions])
        np.savez(f"{output_directory}/eval.npz", positions_np)
