from abc import ABC, abstractmethod
import torch

class AggregationOperator(ABC):
    @abstractmethod
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        pass

class AggregMin(AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            xs = torch.where(~mask, torch.tensor(float('inf'), dtype=xs.dtype, device=xs.device), xs)
        result = torch.amin(xs, dim=dim, keepdim=keepdim)
        return result

class AggregPMean(AggregationOperator):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            # Apply mask to exclude certain values
            xs = xs * mask
            sum_p = torch.sum(xs ** self.p, dim=dim, keepdim=keepdim)
            count_p = torch.sum(mask, dim=dim, keepdim=keepdim)
        else:
            sum_p = torch.sum(xs ** self.p, dim=dim, keepdim=keepdim)
            count_p = xs.size(dim) if dim is not None else xs.numel()
        result = (sum_p / count_p) ** (1 / self.p)
        return result

class AggregPMeanError(AggregationOperator):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            xs = torch.where(~mask, torch.tensor(0.0, dtype=xs.dtype, device=xs.device), xs)
            sum_p = torch.sum((1 - xs) ** self.p, dim=dim, keepdim=keepdim)
            count_p = torch.sum(mask, dim=dim, keepdim=keepdim)
        else:
            sum_p = torch.sum((1 - xs) ** self.p, dim=dim, keepdim=keepdim)
            count_p = xs.size(dim) if dim is not None else xs.numel()
        result = 1 - (sum_p / count_p) ** (1 / self.p)
        return result
    
class SatAgg:
    def __init__(self, agg_op = AggregPMeanError(p=2)):
        if not isinstance(agg_op, AggregationOperator):
            raise TypeError("agg_op must be an instance of AggregationOperator")
        self.agg_op = agg_op

    def __call__(self, *closed_formulas):
        # Collect the truth values from the closed formulas
        truth_values = [torch.tensor(cf, dtype=torch.float32, device=cf.device) if not isinstance(cf, torch.Tensor) else cf for cf in closed_formulas]
        
        # Stack the truth values into a single tensor
        truth_values = torch.stack(truth_values)
        # Apply the aggregation operator
        return self.agg_op(truth_values, dim=0)
