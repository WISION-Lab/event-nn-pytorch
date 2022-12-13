from math import prod

import torch
import torch.nn as nn

from eventnn.utils import Counts


class EventModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.count_mode = False
        self.counts = Counts()
        self.event_mode = False
        self.memory_loss_mode = False
        self.sparse_mode = True

    def accumulators(self):
        return self.modules_of_type(Accumulator)

    def clear_counts(self, recurse=True):
        self.counts.clear()
        if recurse:
            for module in self.event_modules():
                module.clear_counts(recurse=False)

    def conventional(self):
        self.eventful(mode=False)

    def counting(self, mode=True):
        self.set_modes(EventModule, "count_mode", mode)

    def eventful(self, mode=True):
        self.set_modes(EventModule, "event_mode", mode)

    def event_modules(self):
        return self.modules_of_type(EventModule)

    def gates(self):
        return self.modules_of_type(Gate)

    def long_term_memory(self):
        self.memory_loss(mode=False)

    def memory_loss(self, mode=True):
        self.set_modes(EventModule, "memory_loss_mode", mode)

    def modules_of_type(self, module_type):
        return list(filter(lambda x: isinstance(x, module_type), self.modules()))

    def no_counting(self):
        self.counting(mode=False)

    def non_sparse(self):
        self.sparse(mode=False)

    def reset(self, recurse=True):
        if recurse:
            for module in self.event_modules():
                module.reset(recurse=False)

    def set_modes(self, module_type, name, value):
        setattr(self, name, value)
        for module in self.modules_of_type(module_type):
            setattr(module, name, value)

    def sparse(self, mode=True):
        self.set_modes(EventModule, "sparse_mode", mode)

    def total_counts(self):
        return sum(x.counts for x in self.event_modules())


class Accumulator(EventModule):
    def __init__(self):
        super().__init__()
        self.first = True
        self.a = None

    def forward(self, x):
        if not self.event_mode:
            return x
        elif self.first:
            return self._forward_flush(x)
        else:
            return self._forward_event(x)

    def reset(self, recurse=False):
        super().reset(recurse=recurse)
        self.first = True
        self.a = None

    def _forward_event(self, x):
        self.a += x
        if self.count_mode:
            n_updates = x.count_nonzero().item()
            self.counts["event_add"] += n_updates
            self.counts["event_load/store"] += 2 * n_updates
        return self.a

    def _forward_flush(self, x):
        self.first = False
        self.a = x.clone()
        if self.count_mode:
            self.counts["event_load/store"] += prod(x.shape)
        return x


class Gate(EventModule):
    def __init__(self):
        super().__init__()
        self.policy = None
        self.first = True
        self.b = None
        self.d = None

    def forward(self, x):
        if not self.event_mode:
            return x
        elif self.first:
            return self._forward_flush(x)
        else:
            return self._forward_event(x)

    def reset(self, recurse=False):
        super().reset(recurse=recurse)
        self.first = True
        self.b = None
        self.d = None

    def _forward_event(self, x):
        c = x - self.b
        self.d += c
        self.b = x
        p = self.d if (self.policy is None) else self.policy(self.d, c)
        if self.memory_loss_mode:
            self.d[:] = 0.0
        else:
            self.d -= p
        if self.count_mode:
            n_updates = c.count_nonzero().item()
            n_transmissions = p.count_nonzero().item()
            self.counts["event_add"] += 2 * n_updates + n_transmissions
            self.counts["event_load/store"] += 4 * n_updates + 2 * n_transmissions
        return p

    def _forward_flush(self, x):
        self.first = False
        self.b = x
        self.d = torch.zeros_like(x)
        if self.count_mode:
            self.counts["event_load/store"] += 2 * prod(x.shape)
        return x
