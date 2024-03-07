import argparse
import json
import math
import os

import numpy as np
import torch
from tqdm import trange

from gradsim import dflex as df
from utils import export_obj, load_mesh

if __name__ == "__main__":
    # Load arguments and simulation configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", type=str, default=os.path.join("sampledata", "configs", "spot.json"),
                        help="Path to simulation configuration variables")
    args = parser.parse_args()
    device = "cuda:0"
    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)

    # Load simulation mesh
    points, tet_indices = load_mesh(simulation_config["mesh"])
    print(f"Running simulation with {len(points)} particles and {len(tet_indices) // 4} tetrahedra")

    # Get simulation configurations
    r = df.quat_multiply(
        df.quat_from_axis_angle((0.0, 0.0, 1.0), math.pi * 0),
        df.quat_from_axis_angle((1.0, 0.0, 0.0), math.pi * 0),
    )
    position = tuple(simulation_config["position"])
    velocity = tuple(simulation_config["initial_velocity"])
    scale = simulation_config["scale"]
    density = simulation_config["density"]
    k_mu = simulation_config["mu"]
    k_lambda = simulation_config["lambda"]
    k_damp = simulation_config["damp"]

    particle_inv_mass = None

    with torch.no_grad():
        builder_gt = df.sim.ModelBuilder()
        builder_gt.add_soft_mesh(pos=position, rot=r, scale=scale, vel=velocity,
                                 vertices=points, indices=tet_indices, density=density,
                                 k_mu=k_mu, k_lambda=k_lambda, k_damp=k_damp)
        model = builder_gt.finalize("cpu")

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

        # model_gt.particle_inv_mass *= 1 / 50
        particle_inv_mass = model.particle_inv_mass.clone()

        state = model.state()

        sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
        sim_steps = int(simulation_config["sim_duration"] / sim_dt)
        sim_time = 0.0

        # DEBUGGING - SAVE INITIAL STATE
        np.savetxt(f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/debugging_logs/initial_gt_positions.txt",
                   state.q.clone().cpu().detach().numpy())
        # END DEBUGGING

        print(f"Starting centroid: {torch.mean(state.q, dim=0)}")
        positions = []
        render_steps = simulation_config["sim_substeps"]
        integrator = df.sim.SemiImplicitIntegrator()
        for i in trange(0, sim_steps):
            state = integrator.forward(model, state, sim_dt)
            sim_time += sim_dt

            if i % render_steps == 0:
                if torch.sum(torch.isnan(state.q[:10])) != 0:
                    raise RuntimeError(f"Nan Detected at step {i}")

                positions.append(state.q)

        # DEBUGGING - PREDICTED POSITIONS
        positions_np = np.array([p.detach().cpu().numpy() for p in positions])
        path = f"/home/pavlos/Desktop/stuff/Uni-Masters/thesis/debugging_logs/predicted_positions.npz"
        np.savez(path, positions_np)
        # END DEBUGGING

        # Make and save a numpy array of the states (for ease of loading into Blender)
        outdir = "output"
        positions_gt_np = np.array([gt.cpu().numpy() for gt in positions])
        np.savez(os.path.join(outdir, "positions_gt.npz"), positions_gt_np)

        # Export the surface mesh (useful for visualization)
        faces = model.tri_indices
        export_obj(positions_gt_np[0], faces, os.path.join(outdir, "simulation_mesh.obj"))

        # Save ground truth data
        np.savetxt(os.path.join(outdir, "mass_gt.txt"), particle_inv_mass.detach().cpu().numpy())
