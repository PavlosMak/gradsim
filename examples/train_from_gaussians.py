import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

import wandb

from gradsim import dflex as df
from utils import load_pseudo_gt_mesh, export_obj, load_mesh

from scipy.spatial import KDTree


def get_volume(v0, v1, v2, v3):
    v1v0 = v1 - v0
    v2v0 = v2 - v0
    v3v0 = v3 - v0
    return 0.166666 * np.dot(np.cross(v1v0, v2v0), v3v0)


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

    # torch.autograd.set_detect_anomaly(True)  # Uncomment to debug backpropagation

    frame_count = training_config["frame_count"]
    sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
    sim_duration = frame_count / simulation_config["physics_engine_rate"]
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    render_steps = simulation_config["sim_substeps"]
    path_to_exp = f"{training_config['path_to_gt']}/{training_config['exp_name']}"
    path_to_gt = f"{path_to_exp}/gt"

    points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    print(f"Fitting simulation with {len(points)} particles and {len(tet_indices)} tetrahedra")

    # r = df.quat_multiply(
    #     df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 1.0),
    #     df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.0),  # we need to reflect around the x-axis
    # )

    # r = df.quat_identity()
    r = df.quat_from_axis_angle((1.0, 0.0, 0.0), 0.5 * math.pi)

    positions_gt = []
    gt_ground_level = torch.inf
    for i in range(frame_count):
        frame_gt_path = f"{path_to_gt}/gt_{i}.txt"
        frame_positions = torch.tensor(np.loadtxt(frame_gt_path, delimiter="\t"), dtype=torch.float32)
        rotation_matrix = torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
        frame_positions = (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
        frame_min_y = torch.min(frame_positions[:, 1])
        gt_ground_level = min(frame_min_y, gt_ground_level)
        positions_gt.append(frame_positions)

    # Correct point coordinate system
    points = (df.quat_to_matrix(r) @ points.transpose(1, 0)).transpose(1, 0)
    # Correct for ground offset
    gt_ground_offset_np = np.array([0, -gt_ground_level, 0], dtype=np.float32)
    points += gt_ground_offset_np
    gt_ground_offset = torch.tensor(gt_ground_offset_np)
    positions_gt = [p + gt_ground_offset for p in positions_gt]

    # Log ground truth positions
    positions_np = np.array([p.detach().cpu().numpy() for p in positions_gt])
    np.savez(f"{training_config['logdir']}/gt_positions.npz", positions_np)

    # For every point in the simulated mesh we need to find the corresponding point in the GT
    tree = KDTree(points)
    common_indices = []
    for p in positions_gt[0]:
        _, index = tree.query(p)
        common_indices.append(index)

    massmodel = SimpleModel(
        100 * torch.rand(points.shape[0]),
        activation=torch.nn.functional.relu,
    )

    position = tuple((0, 0, 0))  # particles are already aligned with GT
    velocity = tuple(simulation_config["initial_velocity"])
    scale = 1.0
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    optimizer = torch.optim.Adam(massmodel.parameters(), lr=training_config["lr"])
    lossfn = torch.nn.MSELoss()

    run = wandb.init(project="Gaussian Inverse Physics")

    epochs = training_config["epochs"]
    for e in range(epochs):
        builder = df.sim.ModelBuilder()
        builder.add_soft_mesh(pos=position, rot=df.quat_identity(), scale=scale, vel=velocity,
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

        model.particle_inv_mass = massmodel()

        integrator = df.sim.SemiImplicitIntegrator()
        sim_time = 0.0
        state = model.state()

        if e == 0:
            print(f"Starting centroid {torch.mean(state.q, dim=0)}")
            print(f"GT 0 centroid {torch.mean(positions_gt[0], dim=0)}")

        faces = model.tri_indices
        export_obj(state.q.clone().detach().cpu().numpy(), faces,
                   os.path.join(training_config["logdir"], "simulation_mesh.obj"))

        positions = []
        losses = []
        inv_mass_errors = []
        mass_errors = []

        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)

            sim_time += sim_dt

            if i % render_steps == 0:
                positions.append(state.q)

        pred_positions = [p[common_indices] for p in positions]
        loss = sum([lossfn(est, gt) for est, gt in
                    zip(pred_positions[::training_config["compare_every"]],
                        positions_gt[::training_config["compare_every"]])])

        if e == 0:
            positions_np = np.array([p.detach().cpu().numpy() for p in positions])
            np.savez(os.path.join(training_config["logdir"], "unoptimized.npz"), positions_np)

        if e % training_config["logging_interval"] == 0 or e == epochs - 1:
            print(f"Epoch: {(e + 1):03d}/{epochs:03d} - Loss: {loss.item():.5f}")
            wandb.log({"Loss": loss.item()})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())

    # Make and save a numpy array of the states (for ease of loading into Blender)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(os.path.join(training_config["logdir"], "predicted.npz"), positions_np)
