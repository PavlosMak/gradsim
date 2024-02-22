import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from utils import export_obj, load_mesh

LOGGING_INTERVAL = 5

if __name__ == "__main__":

    # Get an argument parser with base-level arguments already filled in.
    dflex_base_parser = get_dflex_base_parser()
    # Create a new parser that inherits these arguments.
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument("--outdir", type=str, default="output",
                        help="Directory to store experiment logs in.")
    parser.add_argument("--mesh", type=str, default=os.path.join("sampledata", "tet", "icosphere.tet"),
                        help="Path to input mesh file (.tet format).")
    parser.add_argument("--sim-config", type=str, default=os.path.join("sampledata", "configs", "spot.json"),
                        help="Path to simulation configuration variables")
    parser.add_argument("--sim-duration", type=float,
                        default=2.0, help="Duration of the simulation episode.")
    parser.add_argument("--physics-engine-rate", type=int, default=90,
                        help="Number of physics engine `steps` per 1 second of simulator time.")
    parser.add_argument("--sim-substeps", type=int, default=32,
                        help="Number of sub-steps to integrate, per 1 `step` of the simulation.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda:0"

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    render_steps = args.sim_substeps

    points, tet_indices = load_mesh(args.mesh)

    print(f"Running simulation with {len(points)} particles and {len(tet_indices)} tetrahedra")

    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    position = tuple(simulation_config["position"])
    velocity = tuple(simulation_config["initial_velocity"])
    scale = simulation_config["scale"]
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    particle_inv_mass_gt = None

    with torch.no_grad():
        builder_gt = df.sim.ModelBuilder()
        builder_gt.add_soft_mesh(
            pos=position, rot=r, scale=scale, vel=velocity,
            vertices=points, indices=tet_indices, density=density,
            k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp
        )

        model_gt = builder_gt.finalize("cpu")

        model_gt.tri_ke = 0.0
        model_gt.tri_ka = 0.0
        model_gt.tri_kd = 0.0
        model_gt.tri_kb = 0.0

        model_gt.contact_ke = 1.0e4
        model_gt.contact_kd = 1.0
        model_gt.contact_kf = 10.0
        model_gt.contact_mu = 0.5

        model_gt.particle_radius = 0.05
        model_gt.ground = True

        # model_gt.particle_inv_mass *= 1 / 50
        particle_inv_mass_gt = model_gt.particle_inv_mass.clone()

        integrator = df.sim.SemiImplicitIntegrator()

        state_gt = model_gt.state()

        faces = model_gt.tri_indices

        sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
        sim_steps = int(args.sim_duration / sim_dt)
        sim_time = 0.0

        positions_gt = []
        for i in trange(0, sim_steps):
            state_gt = integrator.forward(model_gt, state_gt, sim_dt)
            sim_time += sim_dt

            if i % render_steps == 0:
                if torch.sum(torch.isnan(state_gt.q[:10])) != 0:
                    raise RuntimeError(f"Nan Detected at step {i}")

                positions_gt.append(state_gt.q)

        # Make and save a numpy array of the states (for ease of loading into Blender)
        positions_gt_np = np.array([gt.cpu().numpy() for gt in positions_gt])
        np.savez(os.path.join(outdir, "positions_gt.npz"), positions_gt_np)

        # Export the surface mesh (useful for visualization)
        export_obj(positions_gt_np[0], faces, os.path.join(outdir, "simulation_mesh.obj"))

        # Save ground truth data
        np.savetxt(os.path.join(outdir, "mass_gt.txt"), particle_inv_mass_gt.detach().cpu().numpy())
        np.savetxt(os.path.join(outdir, "vertices.txt"), state_gt.q.detach().cpu().numpy())
        np.savetxt(os.path.join(outdir, "face.txt"), faces.detach().cpu().numpy())
