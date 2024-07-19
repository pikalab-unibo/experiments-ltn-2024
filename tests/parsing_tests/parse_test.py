import unittest
import torch
from ltn_imp.parsing.parser import LTNConverter

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

        converter = LTNConverter(predicates, functions, connective_impls=connective_impls, quantifier_impls=quantifier_impls)
        ltn_expr = converter(expr)
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

    def test_less_than_expression(self):
        self._test_expression('man(x,z1) and z1 < 3', value=torch.tensor([1.0]))

    def test_more_than_expression(self):
        self._test_expression('man(x,z1) and (z1 > 1)', value=torch.tensor([1.0]))

    def test_less_than_or_equal_expression(self):
        self._test_expression('man(x,z1) and (z1 <= 2)', value=torch.tensor([1.0]))

    def test_more_than_or_equal_expression(self):
        self._test_expression('man(x,z1) and (z1 >= 2)', value=torch.tensor([1.0]))

    def test_complex_comparison_expression(self):
        self._test_expression('Man(x) and ( man(x,z1) and (z1 > 1))', value=torch.tensor([1.0]))

    def test_flip_less_than_expression(self):
        self._test_expression('man(x,z1) and z1 < 3', value=torch.tensor([1.0]))

    def test_flip_more_than_expression(self):
        self._test_expression('man(x,z1) and (z1 > 1)', value=torch.tensor([1.0]))

    def test_flip_less_than_or_equal_expression(self):
        self._test_expression('man(x,z1) and (z1 <= 2)', value=torch.tensor([1.0]))

    def test_flip_more_than_or_equal_expression(self):
        self._test_expression('man(x,z1) and (z1 >= 2)', value=torch.tensor([1.0]))

    def test_flip_complex_comparison_expression(self):
        self._test_expression('Man(x) and ( man(x,z1) and (z1 > 1))', value=torch.tensor([1.0]))

    def test_arithmetic_operations(self):
        self._test_expression('man(x) + 3', value=torch.tensor([5.0]))
        self._test_expression('man(x) - 1', value=torch.tensor([1.0]))
        self._test_expression('2 * man(x)', value=torch.tensor([4.0]))
        self._test_expression('man(x) / 2', value=torch.tensor([1.0]))

    def test_nested_arithmetic_operations(self):
        self._test_expression('(man(x) + 3) * 2', value=torch.tensor([10.0]))
        self._test_expression('(man(x) - 1) / 2', value=torch.tensor([0.5]))
        self._test_expression('2 * (man(x) + 1)', value=torch.tensor([6.0]))
        self._test_expression('man(x) + 2 + 1', value=torch.tensor([5.0]))
        self._test_expression('1 + 1 + 1 + 1 + 1', value=torch.tensor([5.0]))
        self._test_expression('man(x) + man(x) + man(x)', value=torch.tensor([6.0]))

    def test_arithmetic_with_comparison(self):
        self._test_expression('(man(x) + 2) > 3', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('3 < (man(x) + 2)', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression("5 <= 1 + 1 + 1 + 1 + 1", value=torch.tensor([1.0]))  # True as tensor[1.0]
    
    def test_pemdas_operations(self):
        self._test_expression('2 + 3 * 4', value=torch.tensor([14.0]))  # Multiplication before addition
        self._test_expression('(2 + 3) * 4', value=torch.tensor([20.0]))  # Parentheses first
        self._test_expression('4 / 2 * 2', value=torch.tensor([4.0]))  # Division and multiplication left to right
        self._test_expression('4 * 2 / 2', value=torch.tensor([4.0]))  # Multiplication and division left to right
        self._test_expression('2 + 3 * 4 - 5', value=torch.tensor([9.0]))  # Multiplication before addition and subtraction
        self._test_expression('10 - 2 + 3', value=torch.tensor([11.0]))  # Addition and subtraction left to right
        self._test_expression('10 / (2 + 3)', value=torch.tensor([2.0]))  # Parentheses first

    def test_negative_numbers(self):
        self._test_expression('-1 + 2', value=torch.tensor([1.0]))
        self._test_expression('3 + -1', value=torch.tensor([2.0]))
        self._test_expression('-1 - 2', value=torch.tensor([-3.0]))
        self._test_expression('2 * -3', value=torch.tensor([-6.0]))
        self._test_expression('-2 * -3', value=torch.tensor([6.0]))
        self._test_expression('-(2 + 3)', value=torch.tensor([-5.0]))
        self._test_expression('-(man(x) + 2)', value=torch.tensor([-4.0]))
        self._test_expression('man(x) + -2', value=torch.tensor([0.0]))

    def test_negative_numbers_with_comparison(self):
        self._test_expression('-1 < 0', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('-1 > 0', value=torch.tensor([0.0]))  # False as tensor[0.0]
        self._test_expression('-1 <= -1', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('-1 >= -1', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('-man(x) < 0', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('man(x) > -1', value=torch.tensor([1.0]))  # True as tensor[1.0]
        self._test_expression('-(man(x) + -2 + 0) = 0', value=torch.tensor([1.0]))  # True as tensor[1.0]

if __name__ == '__main__':
    unittest.main()
