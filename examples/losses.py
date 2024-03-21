import numpy as np
from scipy.spatial import KDTree
import torch
from utils import save_positions, plot_histogram


class FixedCorrespondenceDistanceLoss:
    def __init__(self, canonical_gt_positions, positions):
        self.tree = KDTree(canonical_gt_positions)
        self.index_map = {}
        for pi, p in enumerate(positions):
            dist, index = self.tree.query(p)
            self.index_map[pi] = (index, dist)

    def __call__(self, predicted_positions, target_positions):
        total_loss = torch.zeros(1)
        frames = predicted_positions.shape[0]
        for frame in range(frames):
            loss = torch.zeros(1)
            for pi in self.index_map:
                ni, dist = self.index_map[pi]
                loss += (torch.linalg.norm(predicted_positions[frame][pi] - target_positions[frame][ni]) - dist) ** 2
            total_loss += (loss / len(self.index_map))
        return total_loss / frames


class MSECorrespondencesLoss:
    def __init__(self, canonical_gt_positions, positions):
        self.tree = KDTree(canonical_gt_positions)
        self.index_map = {}
        for pi, p in enumerate(positions):
            _, index = self.tree.query(p)
            self.index_map[pi] = index

    def __call__(self, predicted_positions, target_positions):
        total_loss = torch.zeros(1)
        frames = predicted_positions.shape[0]
        for frame in range(frames):
            loss = torch.zeros(1)
            for pi in self.index_map:
                ni = self.index_map[pi]
                loss += torch.sum((predicted_positions[frame][pi] - target_positions[frame][ni]) ** 2)
            total_loss += (loss / len(self.index_map))
        return total_loss / frames


class ClosestOnlyLoss:
    def __init__(self, canonical_gt_positions, positions):
        self.tree = KDTree(canonical_gt_positions)
        self.index_map = {}


        neighbors = []
        mathcing_points = []
        dists = []

        for pi, p in enumerate(positions):
            dist, index = self.tree.query(p)

            if dist <= 0.1:
                neighbors.append(canonical_gt_positions[index])
                dists.append(dist)
                mathcing_points.append(torch.tensor(p))
                self.index_map[pi] = index

        neighbors = torch.stack(neighbors)
        save_positions(neighbors, "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/output/matches.npz")
        save_positions(torch.stack(mathcing_points),
                       "/home/pavlos/Desktop/stuff/Uni-Masters/thesis/gradsim/examples/output/positions.npz")
        # plot_histogram(dists)

    def __call__(self, predicted_positions, target_positions):
        total_loss = torch.zeros(1)
        frames = predicted_positions.shape[0]
        for frame in range(frames):
            loss = torch.zeros(1)
            for pi in self.index_map:
                ni = self.index_map[pi]
                loss += torch.sum((predicted_positions[frame][pi] - target_positions[frame][ni]) ** 2)
            total_loss += (loss / len(self.index_map))
        return total_loss / frames
