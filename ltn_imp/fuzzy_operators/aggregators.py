from abc import ABC, abstractmethod
import torch

class AggregationOperator(ABC):
    """
    Abstract class for aggregation operators.

    Every aggregation operator must inherit from this class
    and implement the `__call__()` method.
    """
    
    @abstractmethod
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        """
        Implements the behavior of the aggregation operator.

        Parameters
        ----------
        xs: torch.Tensor
            The tensor on which the aggregation operator is applied.
        dim: tuple of int, optional
            The dimensions over which to aggregate.
        keepdim: bool, optional
            Whether to retain the reduced dimensions in the output tensor.
        mask: torch.Tensor, optional
            A boolean mask to exclude certain values from aggregation.

        Returns
        -------
        torch.Tensor
            The result of the aggregation.
        """
        pass

class AggregMin(AggregationOperator):
    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            xs = torch.where(~mask, torch.tensor(float('inf'), dtype=xs.dtype), xs)
        return torch.amin(xs, dim=dim, keepdim=keepdim)


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

        return (sum_p / count_p) ** (1 / self.p)

class AggregPMeanError(AggregationOperator):
    def __init__(self, p=2):
        self.p = p

    def __call__(self, xs, dim=None, keepdim=False, mask=None):
        if mask is not None:
            # Apply mask to exclude certain values
            xs = torch.where(~mask, torch.tensor(0.0, dtype=xs.dtype), xs)
            sum_p = torch.sum((1 - xs) ** self.p, dim=dim, keepdim=keepdim)
            count_p = torch.sum(mask, dim=dim, keepdim=keepdim)
        else:
            sum_p = torch.sum((1 - xs) ** self.p, dim=dim, keepdim=keepdim)
            count_p = xs.size(dim) if dim is not None else xs.numel()

        return 1 - (sum_p / count_p) ** (1 / self.p)
    

class SatAgg:
    def __init__(self, agg_op = AggregPMeanError(p=2)):
        if not isinstance(agg_op, AggregationOperator):
            raise TypeError("agg_op must be an instance of AggregationOperator")
        self.agg_op = agg_op

    def __call__(self, *closed_formulas):
        # Collect the truth values from the closed formulas
        truth_values = [torch.tensor(cf, dtype=torch.float32) if not isinstance(cf, torch.Tensor) else cf for cf in closed_formulas]
        
        # Stack the truth values into a single tensor
        truth_values = torch.stack(truth_values)

        # Apply the aggregation operator
        return self.agg_op(truth_values, dim=0)