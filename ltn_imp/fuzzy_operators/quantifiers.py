from abc import ABC, abstractmethod
import torch

# Base class for all quantifiers
class Quantifier(ABC):
    @abstractmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

class ExistsQuantifier(Quantifier):
    def __init__(self, method="max"):
        self.method = method

    def forward(self, *args):
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if self.method == "max":
            return torch.max(torch.stack(tensors))
        elif self.method == "mean":
            return torch.mean(torch.stack(tensors))
        elif self.method == "prod":
            return torch.prod(torch.stack(tensors))
        else:
            raise ValueError(f"Unknown method: {self.method}")

class ForallQuantifier(Quantifier):
    def __init__(self, method="min"):
        self.method = method

    def forward(self, *args):
        tensors = [arg for arg in args if isinstance(arg, torch.Tensor)]
        if self.method == "min":
            return torch.min(torch.stack(tensors))
        elif self.method == "mean":
            return torch.mean(torch.stack(tensors))
        elif self.method == "prod":
            return torch.prod(torch.stack(tensors))
        else:
            raise ValueError(f"Unknown method: {self.method}")