"""Local, allowlisted tools and the model's internal tool protocol."""
import ast
from datetime import datetime
import json
import math
import operator
import re

import torch

ROUTES = ('normal', 'time', 'weather', 'calculate')
TOOLS = frozenset(ROUTES[1:])
OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
TRANSLATIONS = (
    (r'\b(?:multiplied by|times)\b', '*'),
    (r'\bdivided by\b', '/'),
    (r'\b(?:plus|added to)\b', '+'),
    (r'\bminus\b', '-'),
    (r'\b(?:to the power of|raised to)\b', '**'),
    (r'\b(?:modulo|mod)\b', '%'),
    (r'(?<=\d)\s*[xX]\s*(?=[\d(])', '*'),
)
TOOL_RESULT = 'Tool result'


def calculate(expression):
    if not expression or len(expression) > 120 or not re.fullmatch(r'[0-9 .+\-*/%()]+', expression):
        raise ValueError('Enter an arithmetic expression using numbers and operators.')
    tree = ast.parse(expression.strip(), mode='eval')
    if sum(1 for _ in ast.walk(tree)) > 50:
        raise ValueError('The expression is too complex.')

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = node.value
        elif isinstance(node, ast.UnaryOp) and type(node.op) in (ast.UAdd, ast.USub):
            value = visit(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError('Exponents must be between -12 and 12.')
            value = OPERATORS[type(node.op)](left, right)
        else:
            raise ValueError('This expression uses an unsupported operation.')
        if type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1e100:
            raise ValueError('The result is outside the supported range.')
        return value

    return visit(tree.body)


def expression_from_message(text):
    value = text.strip().lower().translate(str.maketrans({'×': '*', '÷': '/', '−': '-'}))
    value = value.replace('^', '**')
    value = re.sub(r'\b(?:calculate|compute|evaluate)\s*(?=\()', '', value)
    percentage = re.search(r'(-?\d+(?:\.\d+)?)\s*(?:%|percent)\s+of\s+(-?\d+(?:\.\d+)?)', value)
    if percentage:
        a, b = percentage.groups()
        return f'({a} / 100) * {b}'
    for pattern, replacement in TRANSLATIONS:
        value = re.sub(pattern, replacement, value)
    candidates = re.findall(r'(?<![\w.])(?:\(\s*)*[-+]?\s*(?:\d|\.\d)[0-9 .+\-*/%()]*', value)
    candidates = [part.strip().rstrip('.').strip() for part in candidates]
    valid = [part for part in candidates if re.search(r'[+*/%]|\d\s*-', part)]
    if len(valid) == 1:
        return valid[0]
    if not valid and len(candidates) == 1 and re.search(r'\b(?:calculate|evaluate|compute)\b', text, re.I):
        return candidates[0]
    raise ValueError('Provide one arithmetic expression, such as (8 + 5) * 2.')


def invoke(name, message, now=None):
    if name not in TOOLS:
        raise ValueError('Unknown tool')
    if name == 'calculate':
        expression = ''
        try:
            expression = expression_from_message(message)
            result = calculate(expression)
            formatted = str(result) if type(result) is int else format(result, '.12g')
            return {'expression': expression, 'result': formatted}
        except ZeroDivisionError:
            return {'expression': expression, 'error': 'Division by zero is undefined.'}
        except (ArithmeticError, SyntaxError, ValueError) as error:
            return {'expression': expression, 'error': str(error) if isinstance(error, ValueError)
                    else 'Enter a valid arithmetic expression.'}
    now = now or datetime.now().astimezone()
    if name == 'time':
        request = message.casefold()
        wants_date = bool(re.search(r'\b(?:date|calendar|weekday|day is it|day is today|day of the week|day and month|month and day)\b', request))
        wants_time = bool(re.search(r'\b(?:time|clock|hour)\b', request))
        return {'time': now.strftime('%H:%M'), 'date': now.strftime('%A, %B %d, %Y'),
                'timezone': now.tzname() or 'local time', 'iso': now.isoformat(),
                'display': 'datetime' if wants_date and wants_time else 'date' if wants_date else 'time'}
    season = ('winter', 'spring', 'summer', 'autumn')[(now.month % 12) // 3]
    return {'season': season}


def user_content(text):
    return text.replace('<|', '< |')


def routing_prompt(message):
    return f'<|user|> Choose local tool for: {user_content(message)} <|ai|>'


def result_prompt(message, name, result):
    payload = json.dumps(result, ensure_ascii=False, separators=(',', ':'))
    return f'<|user|> {user_content(message)}\n{TOOL_RESULT} {name}: {user_content(payload)} <|ai|>'


def grounded_reply(name, result):
    if name == 'calculate':
        return result.get('error') or f"The result is {result['result']}."
    if name == 'time':
        if result.get('display') == 'date':
            return f"Today is {result['date']}."
        if result.get('display') == 'datetime':
            return f"Today is {result['date']}, and it’s {result['time']} {result['timezone']}."
        return f"It’s {result['time']} {result['timezone']}."
    return f"It’s {result['season']}."


def generate_text(model, tokenizer, prefix, count):
    ids = tokenizer.encode(prefix).ids[-model.config.block_size:]
    device = next(model.parameters()).device
    x = torch.tensor([ids], dtype=torch.long, device=device)
    stops = {tokenizer.token_to_id(tag) for tag in ('<|eos|>', '<|user|>')}
    return tokenizer.decode(list(model.generate_stream(
        x, count=count, temperature=0, stop_ids=stops,
    ))).strip()


@torch.inference_mode()
def choose_tool(model, tokenizer, message):
    prompt = routing_prompt(message)
    prefix = tokenizer.encode(prompt).ids
    labels = [tokenizer.encode(f'{prompt} {choice}').ids[len(prefix):] for choice in ROUTES]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError('Tool route labels must each occupy one token')
    device = next(model.parameters()).device
    ids = torch.tensor([prefix[-model.config.block_size:]], dtype=torch.long, device=device)
    logits, _ = model(ids)
    candidates = logits[0, -1, [label[0] for label in labels]]
    probabilities = candidates.softmax(dim=0)
    index = int(probabilities.argmax())
    return ROUTES[index] if index and probabilities[index] >= .55 else None

def reply_with_result(model, tokenizer, message, name, result):
    reply = generate_text(model, tokenizer, result_prompt(message, name, result), 36)
    expected = grounded_reply(name, result)
    # The widget and the final text must agree with the actual tool result.
    return reply if reply == expected else expected
