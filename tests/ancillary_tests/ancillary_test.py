import unittest
import torch
from ltn_imp.parsing.parser import LTNConverter
from ltn_imp.parsing.ancillary_modules import ModuleFactory

class TestModuleFactory(unittest.TestCase):

    def setUp(self):
        self.converter = LTNConverter()
        self.factory = ModuleFactory(converter=self.converter)

    def assertAlmostEqualTensor(self, tensor1, tensor2, delta=0.05):
        self.assertTrue(torch.allclose(tensor1, tensor2, atol=delta), f"{tensor1} not close to {tensor2} within {delta}")

    def _test_expression(self, expr: str, expected_name: str, expected_params: list, expected_value, inputs=None):

        # Test create_module method
        module_class = self.factory.create_module(expr)
        module_instance = module_class()

        # Get the result from the dynamically created module's forward method
        result = module_instance(*inputs)
        self.assertAlmostEqualTensor(result, expected_value)

    def test_simple_addition(self):
        expr = "forall x. (Addition(x,y) <->  x + y)"
        expected_name = "Addition"
        expected_params = ['x', 'y']
        expected_value = torch.tensor(5.0)  # Example expected value
        inputs = [torch.tensor(2.0), torch.tensor(3.0), torch.tensor(5.0)]
        self._test_expression(expr, expected_name, expected_params, expected_value, inputs=inputs)

if __name__ == '__main__':
    unittest.main()
