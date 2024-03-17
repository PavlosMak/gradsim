from scipy.spatial import KDTree
import torch


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


class MSECorrespondencesLoss():
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
