import unittest
import torch
from ltn_imp.parsing.parser import convert_to_ltn
from ltn_imp.fuzzy_operators.aggregators import *

class Man(torch.nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x 
    
class Women(torch.nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x, y):
        return x 
    
class Person(torch.nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, z):
        return z

class man(torch.nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.Tensor([2]) 

predicates = {"Man": Man(), "Women": Women(), "Person" : Person()}
functions = {"man": man()}

x = torch.tensor([[1.0]])
y = torch.tensor([[1.0]])
z = torch.tensor([[1.0]])
w = torch.tensor([[1.0]])

test_data = torch.tensor([[1.0, 1.0, 1.0, 1.0]])

class TestParsing(unittest.TestCase):

    def assertAlmostEqualTensor(self, tensor1, tensor2, delta=0.05):
       self.assertTrue(torch.allclose(tensor1, tensor2, atol=delta), f"{tensor1} not close to {tensor2} within {delta}")

    def _test_expression(self, expr: str, expected: dict = None, value = torch.tensor([1.0]), connective_impls=None, quantifier_impls=None):

        if expected is None: 
            expected = {k: v for k,v in globals().items() if len(k) == 1}

        ltn_expr = convert_to_ltn(expr, predicates, functions, connective_impls=connective_impls, quantifier_impls=quantifier_impls)
        result = ltn_expr(expected)
        self.assertAlmostEqualTensor(result, value)
            
    def test_predicate_man(self):
        self._test_expression('Man(x)')
    
    def test_function_man(self):
        self._test_expression('man(x)', value=torch.tensor([2.0]))

    def test_forall(self):
        methods = ["min", "mean", "prod"]
        for method in methods:
            with self.subTest(method=method):
                self._test_expression('forall x. Man(x)', quantifier_impls={'forall': method})

    def test_and(self):
        implementations = [ "prod", "min", "luk"]
        for impl in implementations:
            with self.subTest(impl=impl):
                self._test_expression('Man(x) and Women(x, y)', connective_impls={'and': impl})
    
    def test_or(self):
        implementations = ["max", "prob_sum", "luk"]
        for impl in implementations:
            with self.subTest(impl=impl):
                self._test_expression('Man(x) or Women(x, y)', connective_impls={'or': impl})

    def test_not(self):
        implementations = ["standard", "godel"]
        for impl in implementations:
            with self.subTest(impl=impl):
                self._test_expression('not Man(x)', value=torch.tensor([0.0]), connective_impls={'not': impl})

    def test_implies(self):
        implementations = ["kleene_dienes", "godel", "reichenbach", "goguen", "luk"]
        for impl in implementations:
            with self.subTest(impl=impl):
                self._test_expression('Man(x) -> Women(x, y)', connective_impls={'implies': impl})

    def test_equiv(self):
        self._test_expression('Man(x) <-> Women(x, y)')

    def test_complex_expression(self):
        self._test_expression('Man(x) and Women(x, y) or Person(x)')

    def test_nested_expressions(self):
        self._test_expression('not (Man(x) and Women(x, y)) or Person(x)')

    def test_exists(self):
        methods = ["max", "mean", "prod"]
        for method in methods:
            with self.subTest(method=method):
                self._test_expression('exists y. Women(x, y)', quantifier_impls={'exists': method})

    def test_nested_quantifiers(self):
        self._test_expression('forall x. exists y. Women(x, y)')

    def test_nested_implies(self):
        self._test_expression('Man(x) -> (Women(x, y) -> Person(x))')

    def test_multiple_variables(self):
        self._test_expression('Man(x) and Women(y, z) -> Person(w)')

    def test_shared_variables(self):
        self._test_expression('Man(x) and Women(z, y) and Person(x)')

    def test_batch_inputs(self):
        self._test_expression('Man(x) and Person(x)', expected={'x':test_data})
    
    def test_pmean_sat_agg(self):
        agg = SatAgg(implementation_name="pmean", p=2)
        result = agg(torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([0.9]))
        expected = torch.tensor([0.7])
        self.assertAlmostEqualTensor(result, expected)

    def test_pmean_error_sat_agg(self):
        agg = SatAgg(implementation_name="pmean_error", p=2)
        result = agg(torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([0.9]))
        expected = torch.tensor([0.6614])
        self.assertAlmostEqualTensor(result, expected)

    def test_prod_sat_agg(self):
        agg = SatAgg(implementation_name="prod")
        result = agg(torch.tensor([0.5]), torch.tensor([0.7]), torch.tensor([0.9]))
        expected = torch.tensor([0.315])
        self.assertAlmostEqualTensor(result, expected)

if __name__ == '__main__':
    unittest.main()