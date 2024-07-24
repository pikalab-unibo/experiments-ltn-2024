import unittest
import torch
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.parsing.expression_transformations import transform
from nltk.sem.logic import Expression
from ltn_imp.parsing.ancillary_modules import ModuleFactory

class TestModuleFactory(unittest.TestCase):

    def setUp(self):
        self.factory = ModuleFactory()

    def assertAlmostEqualTensor(self, tensor1, tensor2, delta=0.05):
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=delta), f"{tensor1} not close to {tensor2} within {delta}")

    def _test_expression(self, expr: str, expected_name: str, expected_params: list, expected_value, inputs=None):
        # Test get_name method
        name = self.factory.get_name(expr)
        self.assertEqual(name, expected_name)

        # Test get_params method
        params = self.factory.get_params(expr)
        self.assertEqual(params, expected_params)

        # Test create_module method
        module_class = self.factory.create_module(expr)
        module_instance = module_class()

        # Get the result from the dynamically created module's forward method
        result = module_instance(*inputs)
        self.assertAlmostEqualTensor(result, expected_value)

    def test_simple_addition(self):
        expr = "forall x y. addition(x, y) <->  x + y"
        expected_name = "addition"
        expected_params = ['x', 'y']
        expected_value = torch.tensor(5.0)  # Example expected value
        inputs = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0)]
        self._test_expression(expr, expected_name, expected_params, expected_value, inputs=inputs)

    def test_bounding_box(self):
        expr = "forall c r. exists t1 b1. bounding_box(c, r, t1, b1) <-> ((t1 = (c - r) / 2) and (b1 = (c + r) / 2))"
        expected_name = "bounding_box"
        expected_params = ['c', 'r', 't1', 'b1']

        expected_value = torch.tensor(1.0)  # Example expected value
        inputs = [torch.tensor(4.0), torch.tensor(2.0), torch.tensor(1.0), torch.tensor(3.0) ]
        self._test_expression(expr, expected_name, expected_params, expected_value, inputs=inputs)

    def test_inside_expression(self):
        expr = "forall t1 b1 t2 b2. inside(t1, b1, t2, b2) <-> (t1 < t2 and b1 < b2)"
        expected_name = "inside"
        expected_params = ['t1', 'b1', 't2', 'b2']
        expected_value = torch.tensor(1.0)  # Example expected value
        inputs = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(3.0), torch.tensor(4.0)]
        self._test_expression(expr, expected_name, expected_params, expected_value, inputs=inputs)

    def test_outside_expression(self):
        expr = "forall t1 b1 t2 b2. outside(t1, b1, t2, b2) <-> (t1 > b2 or b1 > b2)"
        expected_name = "outside"
        expected_params = ['t1', 'b1', 't2', 'b2']
        expected_value = torch.tensor(1.0)  # Example expected value
        inputs = [torch.tensor(5.0), torch.tensor(6.0), torch.tensor(3.0), torch.tensor(4.0)]
        self._test_expression(expr, expected_name, expected_params, expected_value, inputs=inputs)

if __name__ == '__main__':
    unittest.main()
