import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from utils import load_mesh

LOGGING_INTERVAL = 5


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

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # torch.autograd.set_detect_anomaly(True) # Uncomment to debug backpropagation

    device = "cuda:0"

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
    sim_steps = int(args.sim_duration / sim_dt)
    sim_time = 0.0

    render_steps = args.sim_substeps

    points, tet_indices = load_mesh(args.mesh)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    particle_inv_mass_gt = torch.tensor(np.loadtxt(f"{args.datadir}/mass_gt.txt"), dtype=torch.float32)
    positions_gt = np.load(f"{args.datadir}/positions_gt.npz")["arr_0"]
    positions_gt = [torch.tensor(p) for p in positions_gt]

    massmodel = SimpleModel(
        # particle_inv_mass_gt + 50 * torch.rand_like(particle_inv_mass_gt),
        particle_inv_mass_gt + 0.5 * torch.rand_like(particle_inv_mass_gt),
        activation=torch.nn.functional.relu,
    )

    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)

    position = tuple(simulation_config["position"])
    velocity = tuple(simulation_config["initial_velocity"])
    scale = simulation_config["scale"]
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    optimizer = torch.optim.Adam(massmodel.parameters(), lr=1e-1)
    lossfn = torch.nn.MSELoss()

    for e in range(args.epochs):

        builder = df.sim.ModelBuilder()
        builder.add_soft_mesh(
            pos=position,
            rot=r,
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

        activation_strength = 0.3
        activation_penalty = 0.0

        integrator = df.sim.SemiImplicitIntegrator()

        sim_time = 0.0

        state = model.state()

        faces = model.tri_indices
        textures = torch.cat((torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
                              torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
                              torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device)),
                             dim=-1)
        imgs = []
        positions = []
        losses = []
        inv_mass_errors = []
        mass_errors = []
        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)
            sim_time += sim_dt

            if i % render_steps == 0:
                positions.append(state.q)

        loss = sum(
            [lossfn(est, gt) for est, gt in zip(positions[::args.compare_every], positions_gt[::args.compare_every])]
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
