import unittest
import torch
from ltn_imp.fuzzy_operators.connectives import (
    MinAndConnective, ProdAndConnective, LukAndConnective, 
    MaxOrConnective, ProbSumOrConnective, LukOrConnective, 
    StandardNotConnective, GodelNotConnective, 
    KleeneDienesImpliesConnective, GodelImpliesConnective, ReichenbachImpliesConnective, GoguenImpliesConnective, LukImpliesConnective, 
    DefaultIffConnective, DefaultEqConnective
)
from ltn_imp.parsing.parser import LessThan, MoreThan

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

    def test_less_than(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([2, 2, 2], dtype=torch.float32)
        expected = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        self._test_connective(LessThan, a, b, expected)

    def test_more_than(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([2, 2, 2], dtype=torch.float32)
        expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        self._test_connective(MoreThan, a, b, expected)

if __name__ == '__main__':
    unittest.main()
