from abc import ABC, abstractmethod
import torch

# Base class for satisfaction aggregation operators
class SatAggConnective(ABC):
    @abstractmethod
    def aggregate(self, *values):
        pass

    def __call__(self, *values):
        return self.aggregate(*values)

class SatAgg(SatAggConnective):
    def __init__(self, implementation_name="pmean", p=2, stable=True):
        self.p = p
        self.stable = stable

        implementations = {
            "pmean": self.p_mean_agg,
            "pmean_error": self.p_mean_error_agg,
            "prod": self.prod_agg
        }

        if implementation_name not in implementations:
            raise ValueError(f"Unknown implementation: {implementation_name}")

        self.implementation = implementations[implementation_name]

    def aggregate(self, *values):
        return self.implementation(*values)

    def p_mean_agg(self, *values):
        xs = torch.stack(values)
        if self.stable:
            xs = (1 - 1e-4) * xs + 1e-4
        return torch.pow(torch.mean(torch.pow(xs, self.p), dim=0), 1 / self.p)

    def p_mean_error_agg(self, *values):
        xs = torch.stack(values)
        if self.stable:
            xs = (1 - 1e-4) * xs + 1e-4
        errors = 1 - xs
        return 1 - torch.pow(torch.mean(torch.pow(errors, self.p), dim=0), 1 / self.p)

    def prod_agg(self, *values):
        xs = torch.stack(values)
        return torch.prod(xs, dim=0)