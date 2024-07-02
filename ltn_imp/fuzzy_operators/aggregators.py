from abc import ABC, abstractmethod
import torch 

class SatAgg(ABC):
    @abstractmethod
    def aggregate(self, *values):
        pass

    def __call__(self, *values):
        return self.aggregate(*values)

class AvgSatAgg(SatAgg):
    def aggregate(self, *values):
        return torch.mean(torch.stack(values), dim=0)

class ProdSatAgg(SatAgg):
    def aggregate(self, *values):
        return torch.prod(torch.stack(values), dim=0)
