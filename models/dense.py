import torch.nn as nn

from eventnn.core import Accumulator, EventModule, Gate
from eventnn.counted import CountedLinear
from eventnn.utils import Bias


class Dense(EventModule):
    def __init__(self):
        super().__init__()
        self.input = Gate()
        self.hidden_1 = nn.Sequential(
            CountedLinear(in_features=784, out_features=200),
            Accumulator(),
            Bias(features=200),
            nn.ReLU(),
            Gate(),
        )
        self.hidden_2 = nn.Sequential(
            CountedLinear(in_features=200, out_features=200),
            Accumulator(),
            Bias(features=200),
            nn.ReLU(),
            Gate(),
        )
        self.output = nn.Sequential(
            CountedLinear(in_features=200, out_features=10),
            Accumulator(),
            Bias(features=10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.input(x)
        x = self.hidden_1(x)
        x = self.hidden_2(x)
        x = self.output(x)
        return x
