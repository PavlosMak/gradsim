import torch

from pytorch3d.loss import chamfer_distance


def reduction_parser(reduction_string: str):
    """If the string is 'none'/'None' then return None, otherwise return the string."""
    if reduction_string.capitalize() == "None":
        return None
    return reduction_string


def loss_factory(training_config: dict):
    loss_config = training_config["loss"]
    if loss_config["name"] == "chamfer":
        return Chamfer.init_from_config(loss_config)
    elif loss_config["name"] == "mse":
        reduction = reduction_parser(loss_config["reduction"])
        return torch.nn.MSELoss(reduction=reduction)
    else:
        raise ValueError(f"No such loss: {loss_config}")


class Chamfer:
    """
    Small wrapper around the Chamfer distance metric from pytorch3D, so we can
    make a callable object that performs it.
    """

    def __init__(self, batch_reduction: str):
        self.batch_reduction = reduction_parser(batch_reduction)

    def init_from_config(config: dict):
        batch_reduction = config["batch_reduction"]
        return Chamfer(batch_reduction)

    def __call__(self, predicted_positions, target_positions):
        return chamfer_distance(predicted_positions, target_positions, batch_reduction=self.batch_reduction)[0]
