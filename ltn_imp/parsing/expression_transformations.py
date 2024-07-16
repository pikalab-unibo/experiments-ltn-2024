import re

def transform_more(expression_str):
    parts = expression_str.split(">")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"moreThan({first}, {second})"
        return transformed_expression
    else:
        return expression_str

def transform_more_equal(expression_str):
    parts = expression_str.split(">=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"moreThan({first}, {second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str

def transform_less(expression_str):
    parts = expression_str.split("<")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lessThan({first}, {second})"
        return transformed_expression
    else:
        return expression_str

def transform_less_equal(expression_str):
    parts = expression_str.split("<=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lessThan({first}, {second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str

def transform_expression(expression_str):

    patterns = {
        "<=": re.compile(r"(\w+\([^\)]*\)|\w+)\s*<=\s*(\w+\([^\)]*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">=": re.compile(r"(\w+\([^\)]*\)|\w+)\s*>=\s*(\w+\([^\)]*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "<": re.compile(r"(\w+\([^\)]*\)|\w+)\s*<\s*(\w+\([^\)]*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">": re.compile(r"(\w+\([^\)]*\)|\w+)\s*>\s*(\w+\([^\)]*\)|\w+(\.\d+)?|\d+(\.\d+)?)")
    }
    
    for op, pattern in patterns.items():
        matches = pattern.findall(expression_str)
        for match in matches:
            subexpr = f"{match[0]} {op} {match[1]}"
            if op == "<=":
                transformed_subexpr = transform_less_equal(subexpr)
            elif op == ">=":
                transformed_subexpr = transform_more_equal(subexpr)
            elif op == "<":
                transformed_subexpr = transform_less(subexpr)
            elif op == ">":
                transformed_subexpr = transform_more(subexpr)
            expression_str = expression_str.replace(subexpr, transformed_subexpr)
    
    return expression_str