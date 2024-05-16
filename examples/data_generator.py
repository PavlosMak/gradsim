import argparse
import json
import math
import os

import numpy as np
import torch

from examples.training_utils import forward_pass
from gradsim import dflex as df
from utils import export_obj, load_mesh, classical_lame_to_neo

if __name__ == "__main__":
    # Load arguments and simulation configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-config", type=str, default=os.path.join("sampledata", "configs", "spot.json"),
                        help="Path to simulation configuration variables")
    args = parser.parse_args()
    device = "cuda:0"
    with open(args.sim_config) as config_file:
        simulation_config = json.load(config_file)
    outdir = simulation_config["outdir"]

    # Load simulation mesh
    points, tet_indices, _ = load_mesh(simulation_config["mesh"])

    np.savez(f"{outdir}/vertices.npz", points)
    np.savez(f"{outdir}/tets.npz", tet_indices)

    print(f"Running simulation with {len(points)} particles and {len(tet_indices) // 4} tetrahedra")

    # Get simulation configurations
    r = eval(simulation_config["initial_rotation"])
    position = tuple(simulation_config["position"])
    velocity = tuple(simulation_config["initial_velocity"])
    scale = simulation_config["scale"]
    density = simulation_config["density"]
    mu_lame = simulation_config["mu"]
    lambda_lame = simulation_config["lambda"]
    k_mu, k_lambda = classical_lame_to_neo(mu_lame, lambda_lame)
    k_damp = simulation_config["damp"]

    particle_inv_mass = None

    fix_top_plane = False
    if "fix_top_plane" in simulation_config:
        fix_top_plane = simulation_config["fix_top_plane"]

    contact_params=None
    if "contact_params" in simulation_config:
        contact_params = simulation_config["contact_params"]

    print(contact_params)
    with torch.no_grad():
        sim_dt = (1.0 / simulation_config["physics_engine_rate"]) / simulation_config["sim_substeps"]
        sim_steps = int(simulation_config["sim_duration"] / sim_dt)
        sim_time = 0.0
        render_steps = simulation_config["sim_substeps"]

        positions, model, state, average_initial_velocity = forward_pass(position, r,
                                                                         scale, velocity, points, tet_indices, density,
                                                                         k_mu, k_lambda, k_damp,
                                                                         sim_steps, sim_dt, render_steps, fix_top_plane=fix_top_plane, contact_params=contact_params)
    # Output results
    masses = model.particle_inv_mass.detach().cpu().numpy()
    np.savez(f"{outdir}/particle_inv_mass.npz", masses)

    positions_gt_np = np.array([gt.cpu().numpy() for gt in positions])
    np.savez(os.path.join(outdir, "positions_gt.npz"), positions_gt_np)

    # Export the surface mesh (useful for visualization)
    faces = model.tri_indices
    export_obj(positions_gt_np[0], faces, os.path.join(outdir, "simulation_mesh.obj"))
