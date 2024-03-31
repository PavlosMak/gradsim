import numpy as np
import torch
from tqdm import trange

import math
from gradsim import dflex as df

from utils import lame_from_young


# TODO: ADD CITATION TO PACNERF CODEBASE
def constraint(x, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return y_scale * torch.tanh(x_scale * x) + (bound[0] + y_scale)


def constraint_inv(y, bound):
    r = bound[1] - bound[0]
    y_scale = r / 2
    x_scale = 2 / r
    return torch.arctanh((y - (bound[0] + y_scale)) / y_scale) / x_scale


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


# class PhysicalModelYoungPoisson(torch.nn.Module):
#     def __init__(self, initial_E, initial_nu, initial_velocity, initial_masses, update_scale_elasticity=0.1,
#                  update_scale_velocity=0.1, update_scale_masses=0.1):
#         super(PhysicalModelYoungPoisson, self).__init__()
#         self.E_update = torch.nn.Parameter(torch.rand_like(initial_E) * update_scale_elasticity)
#         self.initial_E = initial_E
#
#         self.initial_nu = initial_nu
#         self.nu_update = torch.nn.Parameter(torch.tensor(0.1))
#
#         self.velocity_update = torch.nn.Parameter(torch.rand_like(initial_velocity) * update_scale_velocity)
#         self.initial_velocity = initial_velocity
#
#         self.initial_masses = initial_masses
#         self.mass_update = torch.nn.Parameter(torch.rand_like(initial_masses) * update_scale_masses)
#
#     def forward(self):
#         out_E = torch.nn.functional.relu(self.E_update + self.initial_E) + 1e-8
#         out_nu = torch.clamp(self.nu_update + self.initial_nu, 0, 0.5)
#
#         out_mu, out_lambda = lame_from_young(out_E, out_nu)
#
#         out_velocity = self.velocity_update + self.initial_velocity
#
#         out_masses = torch.nn.functional.relu(self.mass_update + self.initial_masses) + 1e-8
#
#         return out_mu, out_lambda, out_velocity, out_masses

class PhysicalModelYoungPoisson(torch.nn.Module):
    def __init__(self, initial_E, initial_nu, initial_velocity, initial_masses, update_scale_elasticity=0.1,
                 update_scale_velocity=0.1, update_scale_masses=0.1):
        super(PhysicalModelYoungPoisson, self).__init__()
        # self.global_E = torch.nn.Parameter(torch.log10(initial_E))
        self.E_update = torch.nn.Parameter(torch.rand_like(initial_E) * update_scale_elasticity)
        self.initial_E = initial_E
        self.nu_bound = [-0.45, 0.45]
        self.global_nu = torch.nn.Parameter(constraint_inv(torch.tensor(initial_nu), self.nu_bound))

        self.velocity_update = torch.nn.Parameter(torch.rand_like(initial_velocity) * update_scale_velocity)
        self.initial_velocity = initial_velocity

        self.initial_masses = initial_masses
        self.mass_update = torch.nn.Parameter(torch.rand_like(initial_masses) * update_scale_masses)

    def forward(self):
        # out_E = 10 ** torch.nn.functional.relu(self.initial_E) + 1e-8
        out_E = torch.nn.functional.relu(self.E_update + self.initial_E) + 1e-8
        out_nu = constraint(self.global_nu, self.nu_bound)

        out_mu, out_lambda = lame_from_young(out_E, out_nu)

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
                 prediction_model=None, mass_noise=None, fix_top_plane=False,
                 optimization_set={"mu", "lambda", "velocity", "masses"}):
    model = model_factory(position, r, scale, velocity, points, tet_indices, density, k_mu, k_lambda, k_damp)

    if mass_noise is not None:
        model.particle_inv_mass = model.particle_inv_mass + model.particle_inv_mass * mass_noise

    if prediction_model:
        k_mu, k_lambda, velocity, masses = prediction_model()
        if "mu" in optimization_set:
            model.tet_materials[:, 0] = k_mu
        if "lambda" in optimization_set:
            model.tet_materials[:, 1] = k_lambda
        if "velocity" in optimization_set:
            model.particle_v[:, ] = velocity
        if "masses" in optimization_set:
            model.particle_inv_mass = masses

    average_initial_velocity = torch.mean(model.particle_v, dim=0)

    integrator = df.sim.SemiImplicitIntegrator()
    sim_time = 0.0
    state = model.state()

    positions = []

    if fix_top_plane:
        mask = state.q[:, 1] == torch.max(state.q[:, 1])
        original_pos = state.q[mask]
        original_vel = torch.zeros_like(state.u[mask])

    for i in trange(0, sim_steps):
        state = integrator.forward(model, state, sim_dt)

        if fix_top_plane:
            state.q[mask] = original_pos
            state.u[mask] = original_vel

        sim_time += sim_dt

        if i % render_steps == 0:
            # print(f"Frame: {i / render_steps} velocity: {torch.mean(state.u, dim=0)}")
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


def initialize_optimizer_young_poisson(training_config: dict, model: PhysicalModelYoungPoisson):
    # param_groups = [
    #     {'name': 'E', 'params': [model.E_update], 'lr': 500},
    #     {'name': 'nu', 'params': [model.nu_update], 'lr': 0.0001},
    #     {'name': 'velocity', 'params': [model.velocity_update], 'lr': training_config["lr"]["velocity"]},
    #     {'name': 'mass', 'params': [model.mass_update], 'lr': training_config["lr"]["mass"]}
    # ]
    param_groups = [
        {'name': 'E', 'params': [model.E_update], 'lr': 500},
        {'name': 'nu', 'params': [model.global_nu], 'lr': 0.01},
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


def initialize_from_config(training_config, field_name, default_value):
    if field_name in training_config:
        return eval(training_config[field_name])
    return default_value
