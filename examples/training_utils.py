import numpy as np
import torch
from tqdm import trange

import math
from gradsim import dflex as df


class PhysicalModel(torch.nn.Module):
    def __init__(self, initial_mu, initial_lambda, initial_velocity, initial_masses, update_scale_lame=0.1,
                 update_scale_velocity=0.1, update_scale_masses=0.1):
        super(PhysicalModel, self).__init__()
        self.mu_update = torch.nn.Parameter(torch.rand_like(initial_mu) * update_scale_lame)
        self.initial_mu = initial_mu

        self.initial_lambda = initial_lambda
        self.lambda_update = torch.nn.Parameter(torch.rand_like(initial_lambda) * update_scale_lame)

        self.velocity_update = torch.nn.Parameter(torch.rand_like(initial_velocity) * update_scale_velocity)
        self.initial_velocity = initial_velocity

        self.initial_masses = initial_masses
        self.mass_update = torch.nn.Parameter(torch.rand_like(initial_masses) * update_scale_masses)

    def forward(self):
        out_mu = torch.nn.functional.relu(self.mu_update + self.initial_mu) + 1e-8
        out_lambda = torch.nn.functional.relu(self.lambda_update + self.initial_lambda) + 1e-8

        out_velocity = self.velocity_update + self.initial_velocity

        out_masses = torch.nn.functional.relu(self.mass_update + self.initial_masses) + 1e-8

        return out_mu, out_lambda, out_velocity, out_masses


def model_factory(pos, rot, scale, vel, vertices, tet_indices, density, k_mu, k_lambda, k_damp):
    builder = df.sim.ModelBuilder()
    builder.add_soft_mesh(pos=pos, rot=rot, scale=scale, vel=vel,
                          vertices=vertices, indices=tet_indices, density=density,
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

    return model


def forward_pass(position, r, scale, velocity,
                 points, tet_indices, density, k_mu, k_lambda, k_damp, sim_steps, sim_dt, render_steps,
                 prediction_model=None, mass_noise=None):
    model = model_factory(position, r, scale, velocity, points, tet_indices, density, k_mu, k_lambda, k_damp)

    if mass_noise is not None:
        model.particle_inv_mass = model.particle_inv_mass + model.particle_inv_mass * mass_noise

    if prediction_model:
        k_mu, k_lambda, velocity, masses = prediction_model()
        model.tet_materials[:, 0] = k_mu
        model.tet_materials[:, 1] = k_lambda
        model.particle_v[:, ] = velocity
        # model.particle_inv_mass = masses

    average_initial_velocity = torch.mean(model.particle_v, dim=0)

    integrator = df.sim.SemiImplicitIntegrator()
    sim_time = 0.0
    state = model.state()

    positions = []

    for i in trange(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)

        sim_time += sim_dt

        if i % render_steps == 0:
            positions.append(state.q)

    positions = torch.stack(positions)

    return positions, model, state, average_initial_velocity


def initialize_optimizer(training_config: dict, model: PhysicalModel):
    param_groups = [
        {'name': 'mu', 'params': [model.mu_update], 'lr': training_config["lr"]["mu"]},
        {'name': 'lambda', 'params': [model.lambda_update], 'lr': training_config["lr"]["lambda"]},
        {'name': 'velocity', 'params': [model.velocity_update], 'lr': training_config["lr"]["velocity"]},
        {'name': 'mass', 'params': [model.mass_update], 'lr': training_config["lr"]["mass"]}
    ]
    return torch.optim.Adam(param_groups)


def load_gt_positions(training_config: dict):
    r = eval(training_config["gt_rotation"])
    path_to_gt = training_config["path_to_gt"]
    positions_pseudo_gt = np.load(path_to_gt)["arr_0"]
    edited_positions = []
    if "transform_gt_points" in training_config:
        gt_scale = training_config["gt_scale"]
        for i in range(training_config["frame_count"]):
            frame_positions = torch.tensor(positions_pseudo_gt[i], dtype=torch.float32)
            rotation_matrix = torch.tensor(df.quat_to_matrix(r), dtype=torch.float32)
            frame_positions = gt_scale * (rotation_matrix @ frame_positions.transpose(0, 1)).transpose(0, 1)
            edited_positions.append(frame_positions)
        positions_pseudo_gt = torch.stack(edited_positions)
        if "offset_floor" in training_config:
            floor_level_offset = torch.zeros(3)
            floor_level_offset[1] = torch.min(positions_pseudo_gt[:, :, 1].flatten())
            positions_pseudo_gt -= floor_level_offset
    else:
        positions_pseudo_gt = torch.tensor(positions_pseudo_gt)
    return positions_pseudo_gt


def lossfn(predicted_positions, gt_positions, index_map):
    total_loss = torch.zeros(1)
    frames = predicted_positions.shape[0]
    for frame in range(frames):
        loss = torch.zeros(1)
        for pi in index_map:
            ni, dist = index_map[pi]
            loss += (torch.linalg.norm(predicted_positions[frame][pi] - gt_positions[frame][ni]) - dist) ** 2
        total_loss += (loss / len(index_map))
    return total_loss / frames
