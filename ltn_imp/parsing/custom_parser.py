from nltk.sem.logic import LogicParser, LogicalExpressionException, ConstantExpression

class CustomLogicParser(LogicParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, s):
        try:
            return super().__call__(s)
        except LogicalExpressionException:
            return ConstantExpression(s)