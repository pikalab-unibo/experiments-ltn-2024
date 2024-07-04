import unittest
import torch
from ltn_imp.parsing.parser import convert_to_ltn

class Man(torch.nn.Module):
    def forward(self, x):
        return x

class Women(torch.nn.Module):
    def forward(self, x, y):
        return x

class Person(torch.nn.Module):
    def forward(self, z):
        return z

class man(torch.nn.Module):
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

    def _test_expression(self, expr: str, expected: dict = None, value=torch.tensor([1.0]), connective_impls=None, quantifier_impls=None):

        if expected is None: 
            expected = {k: v for k,v in globals().items() if len(k) == 1}

        ltn_expr = convert_to_ltn(expr, predicates, functions, connective_impls=connective_impls, quantifier_impls=quantifier_impls)
        result = ltn_expr(expected)
        self.assertAlmostEqualTensor(result, value)
            
    def test_simple_expression(self):
        self._test_expression('Man(x)')
    
    def test_function_expression(self):
        self._test_expression('man(x)', value=torch.tensor([2.0]))

    def test_batch_expression(self):
        self._test_expression('Man(x) and Person(x)', expected={'x': test_data})

    def test_shared_variables_expression(self):
        self._test_expression('Man(x) and Women(z, y) and Person(x)')

    def test_nested_expression(self):
        self._test_expression('not (Man(x) and Women(x, y)) or Person(x)')

    def test_simple_connectives(self):
        self._test_expression('Man(x) and Women(x, y)', value=torch.tensor([1.0]))

    def test_nested_connectives(self):
        self._test_expression('Man(x) and (Women(x, y) or Person(z))', value=torch.tensor([1.0]))

    def test_simple_quantifiers(self):
        quantifier_impls = {'forall': 'pmean'}
        self._test_expression('forall x. Man(x)', quantifier_impls=quantifier_impls, value=torch.tensor([1.0]))
         
    def test_nested_quantifiers(self):
        quantifier_impls = {'forall': 'min', 'exists': 'pmean'}
        self._test_expression('forall x. (exists y. Man(x) and Women(x, y))', quantifier_impls=quantifier_impls, value=torch.tensor([1.0]))
        self._test_expression('exists x. (forall y. Man(x) and Women(x, y))', quantifier_impls=quantifier_impls, value=torch.tensor([1.0]))

if __name__ == '__main__':
    unittest.main()
