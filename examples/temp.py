import matplotlib.pyplot as plt
import numpy as np
import torch
import json
import math

from utils import lame_from_young

from examples.utils import load_mesh, save_positions
from gradsim import dflex as df

from examples.training_utils import load_gt_positions, forward_pass, constraint


if __name__ == '__main__':
    bound = [-0.45, 0.45]
    xs = torch.linspace(-1, 1, steps=1000)
    ys = [constraint(x, bound) for x in xs]

    plt.plot(xs, ys)
    plt.xlabel("x")
    plt.ylabel("Activation Value")
    plt.show()