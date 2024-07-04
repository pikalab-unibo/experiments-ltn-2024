from abc import ABC, abstractmethod
from ltn_imp.fuzzy_operators.aggregators import *

# Define the available aggregation operators
agg_op_map = {
    "min": AggregMin,
    "pmean": AggregPMean,
    "pmean_error": AggregPMeanError
}

# Abstract class for quantifiers
class Quantifier(ABC):
    def __init__(self, method):
        if method not in agg_op_map:
            raise ValueError(f"Unknown aggregation operator: {method}")
        self.agg_op = agg_op_map[method]()  # Initialize the aggregation operator

    @abstractmethod
    def __call__(self, truth_values, dim=None):
        pass

class ForallQuantifier(Quantifier):
    def __init__(self, method="pmean_error"):
        super().__init__(method)

    def __call__(self, truth_values, dim=None):
        if not isinstance(truth_values, torch.Tensor):
            truth_values = torch.tensor(truth_values, dtype=torch.float32)
        return self.agg_op(truth_values, dim=dim)

class ExistsQuantifier(Quantifier):
    def __init__(self, method="pmean"):
        super().__init__(method)

    def __call__(self, truth_values, dim=None):
        if not isinstance(truth_values, torch.Tensor):
            truth_values = torch.tensor(truth_values, dtype=torch.float32)
        return self.agg_op(truth_values, dim=dim)