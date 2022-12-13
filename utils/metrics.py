import torch


# This class defines (informally) the interface for Metric objects.
class Metric:
    # noinspection PyMethodMayBeStatic
    def compute(self):
        return 0

    def reset(self):
        pass

    def update(self, *args):
        pass


class MeanValue(Metric):
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def compute(self):
        return 0.0 if (self.count == 0) else self.sum / self.count

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value):
        self.sum += value
        self.count += 1


# The PCK metric for human pose estimation
class PCK(Metric):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.n_correct = 0
        self.n_total = 0

    def compute(self):
        return 0.0 if (self.n_total == 0) else self.n_correct / self.n_total

    def reset(self):
        self.n_correct = 0
        self.n_total = 0

    def update(self, true, pred, size):
        w = true[..., 0].amax(dim=-1) - true[..., 0].amin(dim=-1)
        h = true[..., 1].amax(dim=-1) - true[..., 1].amin(dim=-1)
        threshold = self.alpha * torch.maximum(h, w).unsqueeze(dim=-1)
        x = true[..., 0]
        y = true[..., 1]
        x_in_bounds = x.ge(0).logical_and(x.lt(size[1]))
        y_in_bounds = y.ge(0).logical_and(y.lt(size[0]))
        in_bounds = x_in_bounds.logical_and(y_in_bounds)
        self.n_correct += (
            (true - pred).norm(dim=-1).lt(threshold)[in_bounds].sum().item()
        )
        self.n_total += in_bounds.sum().item()
