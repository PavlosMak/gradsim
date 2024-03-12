import torch
from tqdm import trange

from gradsim import dflex as df


class SimpleModel(torch.nn.Module):
    """A thin wrapper around a parameter, for convenient use with optim. """

    def __init__(self, param, activation=None, update_scale=0.1):
        super(SimpleModel, self).__init__()
        self.update = torch.nn.Parameter(torch.rand(param.shape) * update_scale)
        self.param = param
        self.activation = activation

    def forward(self):
        out = self.param + self.update
        if self.activation is not None:
            return self.activation(out) + 1e-8
        return out


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
                 velocity_model=None, k_mu_model=None, k_lambda_model=None, mass_model=None):
    model = model_factory(position, r, scale, velocity, points, tet_indices, density, k_mu, k_lambda, k_damp)

    # if velocity_model:
    #     model.particle_v[:, ] = velocity_model()
    if k_mu_model:
        model.tet_materials[:, 0] = k_mu_model()
    if k_lambda_model:
        model.tet_materials[:, 1] = k_lambda_model()
    # if mass_model:
    #     model.particle_inv_mass = mass_model()
    #
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
