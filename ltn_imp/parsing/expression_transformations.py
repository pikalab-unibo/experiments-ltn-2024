import re

def transform_more(expression_str):
    parts = expression_str.split(">")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"moreThan({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_more_equal(expression_str):
    parts = expression_str.split(">=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"moreThan({first},{second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str

def transform_less(expression_str):
    parts = expression_str.split("<")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lessThan({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_less_equal(expression_str):
    parts = expression_str.split("<=")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"lessThan({first},{second}) or ({first} = {second})"
        return transformed_expression
    else:
        return expression_str
    
def transform_add(expression_str):
    parts = expression_str.split("+")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"add({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_subtract(expression_str):
    parts = expression_str.split("-")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"subtract({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_multiply(expression_str):
    parts = expression_str.split("*")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"multiply({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_divide(expression_str):
    parts = expression_str.split("/")
    if len(parts) == 2:
        first, second = parts[0].strip(), parts[1].strip()
        transformed_expression = f"divide({first},{second})"
        return transformed_expression
    else:
        return expression_str

def transform_encapsulated(expression_str):
    # Updated pattern to ensure the left side of the parentheses is empty or whitespace
    encapsulated_pattern = re.compile(r'(?<!\w)\(([^()]*?(?:\([^()]*\)[^()]*?)*?)\)')
    while encapsulated_pattern.search(expression_str):
        expression_str = encapsulated_pattern.sub(lambda x: transform_expression(x.group(1)), expression_str)
    return expression_str


def transform_expression(expression_str):

    expression_str = transform_encapsulated(expression_str)

    patterns = {
        "*": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*\*\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "/": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*/\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "+": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*\+\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "-": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*-\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "<=": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*<=\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">=": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*>=\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        "<": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*<\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
        ">": re.compile(r"(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+)\s*>\s*(\w+\((?:[^)(]+|\((?:[^)(]+|\([^)(]*\))*\))*\)|\w+(\.\d+)?|\d+(\.\d+)?)"),
    }

    for op, pattern in patterns.items():
        matches = pattern.findall(expression_str)
        for match in matches:
            subexpr = f"{match[0]} {op} {match[1]}"
            if op == "+":
                transformed_subexpr = transform_add(subexpr)
            elif op == "-":
                transformed_subexpr = transform_subtract(subexpr)
            elif op == "*":
                transformed_subexpr = transform_multiply(subexpr)
            elif op == "/":
                transformed_subexpr = transform_divide(subexpr)
            elif op == "<=":
                transformed_subexpr = transform_less_equal(subexpr)
            elif op == ">=":
                transformed_subexpr = transform_more_equal(subexpr)
            elif op == "<":
                transformed_subexpr = transform_less(subexpr)
            elif op == ">":
                transformed_subexpr = transform_more(subexpr)
            expression_str = expression_str.replace(subexpr, transformed_subexpr)

    return expression_str


def transform(expression):

    operators = ["+", "-", "*", "/", "<", "<=", ">", ">="]

    before = expression
    after = transform_expression(expression)
    
    while before != after: # This means that the expression has been transformed but there is a mistake cauising the regex to fail 
        if any(op in after for op in operators):
            before = after 
            after = transform_expression(after)
        else:
            break

    return after