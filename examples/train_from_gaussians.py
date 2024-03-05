import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

import wandb

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from utils import load_pseudo_gt_mesh, export_obj, load_mesh

from scipy.spatial import KDTree

LOGGING_INTERVAL = 5


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

    # Get an argument parser with base-level arguments already filled in.
    dflex_base_parser = get_dflex_base_parser()
    # Create a new parser that inherits these arguments.
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument("--expid", type=str, default="default",
                        help="Unique string identifier for this experiment.", )
    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory to store experiment logs in.")
    parser.add_argument("--mesh", type=str, default=os.path.join("sampledata", "tet", "icosphere.tet"),
                        help="Path to input mesh file (.tet format).")
    parser.add_argument("--sim-duration", type=float,
                        default=2.0, help="Duration of the simulation episode.")
    parser.add_argument("--sim-config", type=str, default=os.path.join("sampledata", "configs", "spot.json"),
                        help="Path to simulation configuration variables")
    parser.add_argument("--physics-engine-rate", type=int, default=90,
                        help="Number of physics engine `steps` per 1 second of simulator time.")
    parser.add_argument("--sim-substeps", type=int, default=32,
                        help="Number of sub-steps to integrate, per 1 `step` of the simulation.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training iterations.")
    parser.add_argument("--lr", type=float, default=1e-1, help="Learning rate.")
    parser.add_argument("--method", type=str, default="gradsim",
                        choices=["noisy-physics-only", "physics-only", "gradsim"],
                        help="Method to use, to optimize for initial velocity.")
    parser.add_argument("--compare-every", type=int, default=1,
                        help="Interval at which video frames are compared.")
    parser.add_argument("--log", action="store_true", help="Log experiment data.")
    parser.add_argument("--datadir", type=str, default="output",
                        help="Directory to store experiment logs in.")
    parser.add_argument("--gt", type=str, help="Directory where the ground truths are saved")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # exp_name = "morning-cloud-226"
    exp_name = "giddy-armadillo-192"
    path_to_exp = f"{args.gt}/{exp_name}"

    # torch.autograd.set_detect_anomaly(True)  # Uncomment to debug backpropagation

    device = "cuda:0"

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # args.sim_substeps = 1000
    frame_count = 13
    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_duration = frame_count / args.physics_engine_rate
    sim_steps = int(sim_duration / sim_dt)
    sim_time = 0.0

    render_steps = args.sim_substeps

    points, tet_indices = load_pseudo_gt_mesh(path_to_exp)
    print(f"Fitting simulation with {len(points)} particles and {len(tet_indices)} tetrahedra")

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi * 1.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 1.0),  # we need to reflect around the x-axis
    )

    particle_inv_mass_gt = torch.tensor(np.loadtxt(f"{args.datadir}/mass_gt.txt"), dtype=torch.float32)

    path_to_gt = f"{path_to_exp}/gt"
    positions_gt = []
    gt_ground_level = 10000000
    for i in range(frame_count):
        frame_gt_path = f"{path_to_gt}/gt_{i}.txt"
        frame_positions = torch.tensor(np.loadtxt(frame_gt_path, delimiter="\t"), dtype=torch.float32)
        frame_positions = frame_positions @ torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
        frame_min_y = torch.min(frame_positions[:, 1])
        gt_ground_level = min(frame_min_y, gt_ground_level)
        positions_gt.append(frame_positions)

    # Correct point coordinate system
    points = np.array([df.quat_to_matrix(r) @ p for p in points], dtype=np.float32)
    # Correct for ground offset
    gt_ground_offset_np = np.array([0, -gt_ground_level, 0], dtype=np.float32)
    points += gt_ground_offset_np
    gt_ground_offset = torch.tensor(gt_ground_offset_np)
    positions_gt = [p + gt_ground_offset for p in positions_gt]

    # DEBUGGING - GT POSITIONS
    positions_np = np.array([p.detach().cpu().numpy() for p in positions_gt])
    path = f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/debugging_logs/full_gt_positions.npz"
    np.savez(path, positions_np)
    # END DEBUGGING

    # For every point in the simulated mesh we need to find the corresponding point in the GT
    tree = KDTree(points)
    common_indices = []
    for p in positions_gt[0]:
        dist, index = tree.query(p)
        common_indices.append(index)

    massmodel = SimpleModel(
        particle_inv_mass_gt + 0.1 * torch.rand_like(particle_inv_mass_gt),
        # 0.5 * torch.rand(points.shape[0]),
        activation=torch.nn.functional.relu,
    )

    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)

    position = tuple((0, 0, 0))  # particles are already aligned with GT
    velocity = tuple(simulation_config["initial_velocity"])
    scale = 1.0
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    learning_rate = args.lr
    optimizer = torch.optim.Adam(massmodel.parameters(), lr=learning_rate)
    lossfn = torch.nn.MSELoss()

    # run = wandb.init(project="Gaussian Inverse Physics")

    epochs = args.epochs
    epochs = 100
    for e in range(epochs):

        builder = df.sim.ModelBuilder()
        builder.add_soft_mesh(
            pos=position,
            rot=df.quat_identity(),
            scale=scale,
            vel=velocity,
            vertices=points,
            indices=tet_indices,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp
        )

        model = builder.finalize("cpu")

        model.tet_kl = 1000.0
        model.tet_km = 1000.0
        model.tet_kd = 1.0

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

        rest_angle = model.edge_rest_angle

        integrator = df.sim.SemiImplicitIntegrator()

        sim_time = 0.0

        state = model.state()

        if e == 0:
            print(f"Starting centroid {torch.mean(state.q, dim=0)}")

        # DEBUGGING - SAVE INITIAL STATE AND FIRST FRAME GT
        np.savetxt(f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/debugging_logs/initial_positions.txt",
                   state.q.clone().cpu().detach().numpy())
        # END DEBUGGING

        faces = model.tri_indices

        positions = []
        losses = []
        inv_mass_errors = []
        mass_errors = []

        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)

            sim_time += sim_dt

            if i % render_steps == 0:
                positions.append(state.q)

        # DEBUGGING - PREDICTED POSITIONS
        positions_np = np.array([p.detach().cpu().numpy() for p in positions])
        path = f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/debugging_logs/predicted_positions.npz"
        np.savez(path, positions_np)
        # END DEBUGGING

        pred_positions = [p[common_indices] for p in positions]
        loss = sum(
            [lossfn(est, gt) for est, gt in
             zip(pred_positions[::args.compare_every], positions_gt[::args.compare_every])]
        )
        inv_mass_err = lossfn(model.particle_inv_mass, particle_inv_mass_gt)
        mass_err = lossfn(
            1 / (model.particle_inv_mass + 1e-6), 1 / (particle_inv_mass_gt + 1e-6)
        )

        if e == 0:
            positions_np = np.array([p.detach().cpu().numpy() for p in positions])
            np.savez(os.path.join(outdir, "unoptimized.npz"), positions_np)

        if e % LOGGING_INTERVAL == 0 or e == args.epochs - 1:
            print(
                f"[EPOCH: {(e + 1):03d}/{args.epochs:03d}] "
                f"Loss: {loss.item():.5f} (Inv) Mass err: {inv_mass_err.item():.5f} "
                f"Mass err: {mass_err.item():.5f}"
            )
            # wandb.log({"Loss": loss.item(), "Inv Mass err": {inv_mass_err.item}, "Mass err": {mass_err.item()}})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        inv_mass_errors.append(inv_mass_err.item())
        mass_errors.append(mass_err.item())

        if (args.log and e % LOGGING_INTERVAL == 0) or (e == args.epochs - 1):
            np.savetxt(os.path.join(outdir, f"mass_{e:05d}.txt"),
                       model.particle_inv_mass.detach().cpu().numpy())

    # Make and save a numpy array of the states (for ease of loading into Blender)
    positions_np = np.array([p.detach().cpu().numpy() for p in positions])
    np.savez(os.path.join(outdir, "predicted.npz"), positions_np)

if args.log:
    np.savetxt(os.path.join(outdir, "losses.txt"), losses)
    np.savetxt(os.path.join(outdir, "inv_mass_errors.txt"), inv_mass_errors)
    np.savetxt(os.path.join(outdir, "mass_errors.txt"), mass_errors)
