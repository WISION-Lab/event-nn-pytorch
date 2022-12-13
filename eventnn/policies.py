from eventnn.core import EventModule


class ConstantThreshold(EventModule):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, x, c):
        if self.count_mode:
            self.counts["event_add"] += c.count_nonzero().item()
        return (x.abs() > self.threshold) * x
