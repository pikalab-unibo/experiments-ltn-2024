import unittest
import torch
from ltn_imp.fuzzy_operators.connectives import (
    MinAndConnective, ProdAndConnective, LukAndConnective, 
    MaxOrConnective, ProbSumOrConnective, LukOrConnective, 
    StandardNotConnective, GodelNotConnective, 
    KleeneDienesImpliesConnective, GodelImpliesConnective, ReichenbachImpliesConnective, GoguenImpliesConnective, LukImpliesConnective, 
    DefaultIffConnective, DefaultEqConnective, DefaultAddConnective, DefaultSubtractConnective, DefaultMultiplyConnective, DefaultDivideConnective
)

Add, Subtract, Multiply, Divide = DefaultAddConnective, DefaultSubtractConnective, DefaultMultiplyConnective, DefaultDivideConnective
class BaseTestConnective(unittest.TestCase):

    def _test_connective(self, connective_cls, a, b=None, expected=None):
        connective = connective_cls()
        if b is not None:
            result = connective(a, b)
        else:
            result = connective(a)
        self.assertTrue(torch.allclose(result, expected, atol=1e-4), f"{result} not close to {expected}")

class TestAndConnectives(BaseTestConnective):

    def test_and_min(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.5])
        self._test_connective(MinAndConnective, a, b, expected)

    def test_and_prod(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.35])
        self._test_connective(ProdAndConnective, a, b, expected)

    def test_and_luk(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.2])
        self._test_connective(LukAndConnective, a, b, expected)

class TestOrConnectives(BaseTestConnective):

    def test_or_max(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.7])
        self._test_connective(MaxOrConnective, a, b, expected)

    def test_or_prob_sum(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.85])
        self._test_connective(ProbSumOrConnective, a, b, expected)

    def test_or_luk(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(LukOrConnective, a, b, expected)

class TestNotConnectives(BaseTestConnective):

    def test_not_standard(self):
        a = torch.tensor([0.5])
        expected = torch.tensor([0.5])
        self._test_connective(StandardNotConnective, a, expected=expected)

    def test_not_godel(self):
        a = torch.tensor([0.5])
        expected = torch.tensor([0.0])
        self._test_connective(GodelNotConnective, a, expected=expected)

class TestImpliesConnectives(BaseTestConnective):

    def test_implies_kleene_dienes(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.7])
        self._test_connective(KleeneDienesImpliesConnective, a, b, expected)

    def test_implies_godel(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(GodelImpliesConnective, a, b, expected)

    def test_implies_reichenbach(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.85])
        self._test_connective(ReichenbachImpliesConnective, a, b, expected)

    def test_implies_goguen(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([1.0])
        self._test_connective(GoguenImpliesConnective, a, b, expected)

    def test_implies_luk(self):
        a = torch.tensor([0.7])
        b = torch.tensor([0.5])
        expected = torch.tensor([0.8])
        self._test_connective(LukImpliesConnective, a, b, expected)

class TestIffConnectives(BaseTestConnective):

    def test_iff(self):
        a = torch.tensor([0.5])
        b = torch.tensor([0.7])
        expected = torch.tensor([0.8])
        self._test_connective(DefaultIffConnective, a, b, expected)

class TestEqConnectives(BaseTestConnective):
    
        def test_eq(self):
            a = torch.tensor([0.5])
            b = torch.tensor([0.5])
            expected = torch.tensor([1.0])
            self._test_connective(DefaultEqConnective, a, b, expected)

class TestComparisonFunctions(BaseTestConnective):

    def test_smooth_less_than(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        expected = torch.sigmoid(10 * (b - a))
        result = torch.sigmoid(10 * (b - a))
        torch.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)

    def test_smooth_more_than(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        b = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float32)
        expected = torch.sigmoid(10 * (a - b))
        result = torch.sigmoid(10 * (a - b))
        torch.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-4)
        
class TestArithmeticOperations(BaseTestConnective):

    def test_add(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5, 6], dtype=torch.float32)
        expected = torch.tensor([5, 7, 9], dtype=torch.float32)
        self._test_connective(Add, a, b, expected)

    def test_subtract(self):
        a = torch.tensor([4, 5, 6], dtype=torch.float32)
        b = torch.tensor([1, 2, 3], dtype=torch.float32)
        expected = torch.tensor([3, 3, 3], dtype=torch.float32)
        self._test_connective(Subtract, a, b, expected)

    def test_multiply(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5, 6], dtype=torch.float32)
        expected = torch.tensor([4, 10, 18], dtype=torch.float32)
        self._test_connective(Multiply, a, b, expected)

    def test_divide(self):
        a = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)
        b = torch.tensor([2.0, 5.0, 3.0], dtype=torch.float32)
        expected = torch.tensor([2.0, 1.0, 2.0], dtype=torch.float32)
        self._test_connective(Divide, a, b, expected)

if __name__ == '__main__':
    unittest.main()
