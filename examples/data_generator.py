import argparse
import json
import math
import os

import kaolin.io.obj
import numpy as np
import torch
from tqdm import trange

from argparsers import get_dflex_base_parser
from gradsim import dflex as df
from utils import read_tet_mesh, export_obj, tetrahedralize

LOGGING_INTERVAL = 5

if __name__ == "__main__":

    # Get an argument parser with base-level arguments already filled in.
    dflex_base_parser = get_dflex_base_parser()
    # Create a new parser that inherits these arguments.
    parser = argparse.ArgumentParser(
        parents=[dflex_base_parser], conflict_handler="resolve"
    )

    parser.add_argument("--expid", type=str, default="default",
                        help="Unique string identifier for this experiment.", )
    parser.add_argument("--outdir", type=str, default=os.path.join("cache", "demo-fem"),
                        help="Directory to store experiment logs in.")
    parser.add_argument("--mesh", type=str, default=os.path.join("sampledata", "tet", "icosphere.tet"),
                        help="Path to input mesh file (.tet format).")
    parser.add_argument("--sim-duration", type=float,
                        default=2.0, help="Duration of the simulation episode.")
    parser.add_argument("--physics-engine-rate", type=int, default=60,
                        help="Number of physics engine `steps` per 1 second of simulator time.")
    parser.add_argument("--sim-substeps", type=int, default=32,
                        help="Number of sub-steps to integrate, per 1 `step` of the simulation.")

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    torch.autograd.set_detect_anomaly(True)

    device = "cuda:0"

    outdir = os.path.join(args.outdir, args.expid)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    render_steps = args.sim_substeps

    phase_count = 8
    phase_step = math.pi / phase_count * 2.0
    phase_freq = 2.5

    if args.mesh.endswith(".tet"):
        points, tet_indices = read_tet_mesh(args.mesh)
    else:
        # TODO: Here we need to load in an obj and tetrahedralize it
        mesh = kaolin.io.obj.import_mesh(args.mesh)
        points, tet_indices = tetrahedralize(mesh.vertices, mesh.faces)
    print(f"Running simulation with {len(points)} particles and {len(tet_indices)} tetrahedra")

    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0.0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0.0),
    )

    vx_init = (1.0 - 3.0) * torch.rand(1) + 3.0
    # pos = torch.tensor([0.0, 2.0, 0.0])
    # vel = torch.tensor([1.5, 0.0, 0.0])

    imgs_gt = []

    particle_inv_mass_gt = None

    with torch.no_grad():
        builder_gt = df.sim.ModelBuilder()
        builder_gt.add_soft_mesh(
            pos=(-2.0, 2.0, 0.0), rot=r,
            scale=1.0, vel=(vx_init.item(), 0.0, 0.0),
            vertices=points, indices=tet_indices, density=10.0)

        model_gt = builder_gt.finalize("cpu")

        model_gt.tet_kl = 1000.0
        model_gt.tet_km = 1000.0
        model_gt.tet_kd = 1.0

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

        particle_inv_mass_gt = model_gt.particle_inv_mass.clone()

        rest_angle = model_gt.edge_rest_angle

        activation_strength = 0.3
        activation_penalty = 0.0

        integrator = df.sim.SemiImplicitIntegrator()

        state_gt = model_gt.state()

        faces = model_gt.tri_indices
        textures = torch.cat((
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.ones(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device),
            torch.zeros(1, faces.shape[-2], 2, 1, dtype=torch.float32, device=device)
        ), dim=-1)

        sim_dt = (1.0 / args.physics_engine_rate) / args.sim_substeps
        sim_steps = int(args.sim_duration / sim_dt)
        sim_time = 0.0

        positions_gt = []
        for i in trange(0, sim_steps):
            state_gt = integrator.forward(model_gt, state_gt, sim_dt)
            sim_time += sim_dt

            if i % render_steps == 0:
                positions_gt.append(state_gt.q)

        # Make and save a numpy array (for ease of loading into blender)
        positions_gt_np = np.array([gt.cpu().numpy() for gt in positions_gt])
        np.savez(os.path.join(outdir, "positions.npz"), positions_gt_np)
        export_obj(positions_gt_np[0], faces, os.path.join(outdir, "simulation_mesh.obj"))

        np.savetxt(os.path.join(outdir, "mass_gt.txt"), particle_inv_mass_gt.detach().cpu().numpy())
        np.savetxt(os.path.join(outdir, "vertices.txt"), state_gt.q.detach().cpu().numpy())
        np.savetxt(os.path.join(outdir, "face.txt"), faces.detach().cpu().numpy())
