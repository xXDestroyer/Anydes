from __future__ import annotations

from dataclasses import dataclass
import html
import re
import unicodedata


_VISUAL_COMMANDS = ("qquad", "quad", "left", "right", "displaystyle", "textstyle")
_KNOWN_IDENTIFIERS = {
    "sqrt", "frac", "vec", "vector", "sum", "integral", "dot", "cross",
    "alpha", "beta", "gamma", "delta", "theta", "lambda", "mu", "pi",
    "rho", "sigma", "phi", "omega",
}
_GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta",
    "Delta": "delta", "theta": "theta", "lambda": "lambda", "mu": "mu",
    "pi": "pi", "rho": "rho", "sigma": "sigma", "Sigma": "sumatoria",
    "phi": "fi", "omega": "omega", "Omega": "omega",
}
_CHAT_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "Sigma": "Σ", "phi": "φ", "omega": "ω", "Omega": "Ω",
}
_SUPERSCRIPT = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")
_SUBSCRIPT = str.maketrans(
    "0123456789+-=()aeijklmnoprstuvxy",
    "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓᵧ",
)
_FORBIDDEN_SPEECH = re.compile(
    r"\\(?:qquad|quad|left|right|begin|end|displaystyle|textstyle)\b|[\\^_{}\[\]]"
)


@dataclass(frozen=True)
class MathPresentation:
    source_text: str
    chat_text: str
    speech_text: str
    math_regions: int = 0
    fallback_used: bool = False


def present_math(text: str) -> MathPresentation:
    source = str(text or "")
    try:
        chat = normalize_chat(source)
        speech, count = normalize_speech(source)
        if not speech.strip():
            raise ValueError("empty_math_speech")
        return MathPresentation(source, chat, speech, count, False)
    except Exception:
        return MathPresentation(
            source,
            _safe_plain_text(source),
            safe_speech_fallback(source),
            0,
            True,
        )


def normalize_chat(text: str) -> str:
    value = _clean_controls(html.unescape(str(text or "")))
    value = _replace_latex_spacing(value, "·")
    value = _replace_matrices(value, chat=True)
    value = _replace_two_group_command(value, "frac", lambda a, b: f"({a}) / ({b})")
    value = _replace_one_group_command(value, "sqrt", lambda a: f"√({a})")
    value = _replace_one_group_command(value, "vec", lambda a: f"vector {a}")
    value = _replace_one_group_command(value, "mathbf", lambda a: a)
    value = value.replace("\\sum", "Σ").replace("\\int", "∫")
    value = value.replace("\\cdot", "·").replace("\\times", "×")
    for command, symbol in _CHAT_GREEK.items():
        value = re.sub(rf"\\{re.escape(command)}\b", symbol, value)
    for command in _VISUAL_COMMANDS:
        value = re.sub(rf"\\{command}\b", " ", value)
    value = value.replace("\\[", "\n").replace("\\]", "\n")
    value = value.replace("\\(", "").replace("\\)", "")
    value = re.sub(r"(?m)^\s*\[\s*$", "", value)
    value = re.sub(r"(?m)^\s*\]\s*$", "", value)
    value = re.sub(r"\^\{([^{}]+)\}", lambda m: _script(m.group(1), True), value)
    value = re.sub(r"\^([0-9n])\b", lambda m: _script(m.group(1), True), value)
    value = re.sub(r"_\{([^{}]+)\}", lambda m: _script(m.group(1), False), value)
    value = re.sub(r"_([0-9]+)", lambda m: _script(m.group(1), False), value)
    value = re.sub(r"_([A-Za-z])", lambda m: _script(m.group(1), False), value)
    value = value.replace("<=", "≤").replace(">=", "≥").replace("!=", "≠")
    value = re.sub(r"\\[A-Za-z]+", "", value)
    value = value.replace("\\", "")
    value = value.replace("{", "").replace("}", "")
    value = "\n".join(_chat_implicit_products(line) for line in value.splitlines())
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def normalize_speech(text: str) -> tuple[str, int]:
    value = _clean_controls(html.unescape(str(text or "")))
    value = re.sub(r"```(?:[A-Za-z0-9_+-]+)?\s*([\s\S]*?)```", r"\1", value)
    value = re.sub(r"`([^`]+)`", r"\1", value)
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = _replace_matrices(value, chat=False)
    regions = 0

    def explicit(match: re.Match[str]) -> str:
        nonlocal regions
        regions += 1
        return " " + verbalize_formula(match.group(1)) + " "

    patterns = (
        r"\\\[([\s\S]*?)\\\]",
        r"\\\(([\s\S]*?)\\\)",
        r"\$\$([\s\S]*?)\$\$",
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
    )
    for pattern in patterns:
        value = re.sub(pattern, explicit, value)

    lines: list[str] = []
    for line in value.splitlines():
        stripped = line.strip()
        if stripped in {"[", "]"}:
            continue
        candidate = stripped.strip("[] ")
        if _looks_like_formula(candidate):
            regions += 1
            lines.append(verbalize_formula(candidate))
        else:
            lines.append(_verbalize_inline_math(line))
    value = "\n".join(lines)
    value = _strip_markdown(value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = value.replace("\\", " ").replace("{", " ").replace("}", " ")
    value = re.sub(r"[\[\]]", " ", value)
    value = value.replace("^", " elevado a ").replace("_", " sub ")
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\s*\n\s*", ". ", value)
    value = re.sub(r"\.{2,}", ".", value)
    value = re.sub(r"\s+([,.;:!?])", r"\1", value)
    value = value.strip(" .")
    if _FORBIDDEN_SPEECH.search(value):
        raise ValueError("unsafe_math_markup_after_normalization")
    return value, regions


def verbalize_formula(formula: str) -> str:
    value = str(formula or "").strip().strip("[]")
    value = _replace_latex_spacing(value, "*")
    value = _replace_top_level_product_separators(value, "*")
    value = _replace_two_group_command(value, "frac", lambda a, b: f"({a})/({b})")
    value = _replace_one_group_command(value, "sqrt", lambda a: f"sqrt({a})")
    value = _replace_one_group_command(value, "vec", lambda a: f"vec({a})")
    value = _replace_one_group_command(value, "mathbf", lambda a: a)
    value = value.replace("\\sum", "sum").replace("\\int", "integral")
    for command, spoken in _GREEK.items():
        value = re.sub(rf"\\{re.escape(command)}\b", spoken, value)
    for command in _VISUAL_COMMANDS:
        value = re.sub(rf"\\{command}\b", " ", value)
    value = re.sub(r"\\(?:begin|end)\s*\{[^{}]*\}", " ", value)
    value = value.replace("\\cdot", " * " if "=" in value else " · ")
    value = value.replace("\\times", " × ")
    value = value.replace("\\leq", " <= ").replace("\\geq", " >= ").replace("\\neq", " != ")
    value = value.replace("√", "sqrt")
    try:
        tokens = _tokenize(value)
        parser = _FormulaParser(tokens)
        node = parser.parse()
        if parser.position != len(tokens):
            raise ValueError("unconsumed_formula_tokens")
        spoken = _speak_node(node)
    except Exception:
        spoken = _token_fallback(value)
    spoken = re.sub(r"\s+", " ", spoken).strip(" ,.;")
    return spoken or "expresión matemática"


def safe_speech_fallback(text: str) -> str:
    value = _clean_controls(html.unescape(str(text or "")))
    value = _replace_latex_spacing(value, "*")
    value = _replace_top_level_product_separators(value, "*")
    value = _replace_two_group_command(value, "frac", lambda a, b: f" {a} dividido para {b} ")
    value = _replace_one_group_command(value, "sqrt", lambda a: f" raíz cuadrada de {a} ")
    value = re.sub(r"\\(?:qquad|quad|left|right|begin|end|displaystyle|textstyle)\b(?:\{[^{}]*\})?", " ", value)
    value = value.replace("<=", " menor o igual que ").replace(">=", " mayor o igual que ")
    value = value.replace("!=", " distinto de ").replace("=", " igual a ")
    value = value.replace("+", " más ").replace("-", " menos ")
    value = value.replace("*", " por ").replace("/", " dividido para ")
    value = re.sub(r"\^\{?2\}?", " al cuadrado ", value)
    value = re.sub(r"\^\{?3\}?", " al cubo ", value)
    value = re.sub(r"\^\{?([^\s{}]+)\}?", r" elevado a \1 ", value)
    value = re.sub(r"_\{([^{}]+)\}", r" sub \1 ", value)
    value = re.sub(r"_([0-9]+|[A-Za-z])", r" sub \1 ", value)
    value = re.sub(r"\\[A-Za-z]+", " ", value)
    value = re.sub(r"[\\{}\[\]()]", " ", value)
    value = _strip_markdown(value)
    return re.sub(r"\s+", " ", value).strip(" .") or "No pude preparar esa fórmula para voz"


def _verbalize_inline_math(line: str) -> str:
    value = line
    patterns = (
        r"(?<!\w)([A-Za-zπθΔ]\s*_\{?[A-Za-z0-9]+\}?)(?!\w)",
        r"(?<!\w)(\([^\n()]{1,40}\)\s*\^\s*\{?[A-Za-z0-9]+\}?)",
        r"(?<!\w)([A-Za-zπθΔ]\s*\^\s*\{?[A-Za-z0-9]+\}?)",
        r"(?<!\w)([A-Za-z]\s*=\s*(?:sqrt\s*\([^\n]{1,120}\)|[^,.;!?\n]{1,120}))",
    )
    for pattern in patterns:
        value = re.sub(pattern, lambda m: verbalize_formula(m.group(1)), value)
    if "\\" in value:
        value = safe_speech_fallback(value)
    return value


def _looks_like_formula(value: str) -> bool:
    if not value or len(value) > 500:
        return False
    operators = len(re.findall(r"(?:\^|_|=|<=|>=|!=|\\frac|\\sqrt|\\sum|\\int|\\vec|\\cdot|\\times|√|Σ|∫)", value))
    words = len(re.findall(r"\b[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]{3,}\b", value))
    implicit_numeric = bool(re.fullmatch(r"\s*\d+(?:[.,]\d+)?[A-Za-z]{1,4}\s*", value))
    return implicit_numeric or (operators >= 1 and (words <= 6 or operators >= 2))


def _replace_latex_spacing(text: str, product: str) -> str:
    """Remove TeX spacing commands, preserving multiplication when they join atoms."""
    pattern = re.compile(r"\\+[,;:!]")

    def replace(match: re.Match[str]) -> str:
        left = text[match.start() - 1] if match.start() else ""
        cursor = match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        right = text[cursor] if cursor < len(text) else ""
        if _atom_end(left) and _atom_start(right):
            return product
        return " "

    return pattern.sub(replace, text)


def _replace_top_level_product_separators(text: str, product: str) -> str:
    """Interpret comma/semicolon as multiplication only at formula top level."""
    chars = list(text)
    depth = 0
    for index, char in enumerate(chars):
        if char in "([{":
            depth += 1
            continue
        if char in ")]}":
            depth = max(0, depth - 1)
            continue
        if depth or char not in ",;":
            continue
        left = next((chars[pos] for pos in range(index - 1, -1, -1) if not chars[pos].isspace()), "")
        right = next((chars[pos] for pos in range(index + 1, len(chars)) if not chars[pos].isspace()), "")
        following = "".join(chars[index + 1:])
        next_separator = min(
            (position for position in (following.find(","), following.find(";")) if position >= 0),
            default=len(following),
        )
        starts_another_equation = "=" in following[:next_separator]
        if _atom_end(left) and _atom_start(right) and not starts_another_equation and not (left.isdigit() and right.isdigit()):
            chars[index] = product
    return "".join(chars)


def _atom_end(char: str) -> bool:
    return bool(char) and (char.isalnum() or char in ")]}")


def _atom_start(char: str) -> bool:
    return bool(char) and (char.isalnum() or char in "([{√")


def _chat_implicit_products(line: str) -> str:
    if not _looks_like_formula(line.strip()):
        return line
    value = _replace_top_level_product_separators(line, "·")
    if value.count("=") > 1:
        return value
    def split_identifier(match: re.Match[str]) -> str:
        identifier = match.group(0)
        if identifier.casefold() in _KNOWN_IDENTIFIERS or len(identifier) > 4:
            return identifier
        return "·".join(identifier)

    value = re.sub(r"(?<=\d)(?=[A-Za-z])", "·", value)
    if "=" in value:
        left, right = value.split("=", 1)
    else:
        left, right = "", value
    right = re.sub(r"(?<![A-Za-z])[A-Za-z]{2,4}(?![A-Za-z])", split_identifier, right)
    right = re.sub(r"(?<=[A-Za-z0-9_)])\s+(?=[A-Za-z0-9_(])", "·", right)
    right = re.sub(r"\s*([·×])\s*", r"\1", right)
    return (left + " = " if left else "") + right.strip()


def _replace_matrices(text: str, *, chat: bool) -> str:
    pattern = re.compile(r"\\begin\{(?:bmatrix|pmatrix|matrix)\}([\s\S]*?)\\end\{(?:bmatrix|pmatrix|matrix)\}")

    def replace(match: re.Match[str]) -> str:
        rows = [row.strip() for row in re.split(r"\\\\", match.group(1)) if row.strip()]
        cells = [[cell.strip() for cell in row.split("&")] for row in rows]
        if chat:
            return "[" + "; ".join("  ".join(row) for row in cells) + "]"
        return "matriz con filas " + "; ".join(", ".join(row) for row in cells)

    return pattern.sub(replace, text)


def _replace_one_group_command(text: str, command: str, render) -> str:
    marker = "\\" + command
    cursor = 0
    output: list[str] = []
    while True:
        index = text.find(marker, cursor)
        if index < 0:
            output.append(text[cursor:])
            return "".join(output)
        output.append(text[cursor:index])
        start = index + len(marker)
        while start < len(text) and text[start].isspace():
            start += 1
        if start >= len(text) or text[start] != "{":
            output.append(marker)
            cursor = start
            continue
        content, end = _take_group(text, start)
        output.append(render(_replace_one_group_command(content, command, render)))
        cursor = end


def _replace_two_group_command(text: str, command: str, render) -> str:
    marker = "\\" + command
    cursor = 0
    output: list[str] = []
    while True:
        index = text.find(marker, cursor)
        if index < 0:
            output.append(text[cursor:])
            return "".join(output)
        output.append(text[cursor:index])
        first_start = index + len(marker)
        while first_start < len(text) and text[first_start].isspace():
            first_start += 1
        if first_start >= len(text) or text[first_start] != "{":
            output.append(marker)
            cursor = first_start
            continue
        first, first_end = _take_group(text, first_start)
        second_start = first_end
        while second_start < len(text) and text[second_start].isspace():
            second_start += 1
        if second_start >= len(text) or text[second_start] != "{":
            output.append(text[index:first_end])
            cursor = first_end
            continue
        second, second_end = _take_group(text, second_start)
        output.append(render(first, second))
        cursor = second_end


def _take_group(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:index], index + 1
    raise ValueError("unclosed_math_group")


def _script(value: str, superscript: bool) -> str:
    table = _SUPERSCRIPT if superscript else _SUBSCRIPT
    translated = value.translate(table)
    return translated if len(translated) == len(value) else ("^(" + value + ")" if superscript else "_(" + value + ")")


def _clean_controls(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(char for char in normalized if char in "\n\t" or unicodedata.category(char) != "Cc")


def _strip_markdown(text: str) -> str:
    value = re.sub(r"(?m)^\s{0,3}(?:#{1,6}|[-*+]\s+|>\s*)", "", text)
    value = re.sub(r"(?:\*\*|__)(.+?)(?:\*\*|__)", r"\1", value)
    value = re.sub(r"(?<!\w)[*_]([^\n*_]+?)[*_](?!\w)", r"\1", value)
    return value


def _safe_plain_text(text: str) -> str:
    value = re.sub(r"\\(?:qquad|quad|left|right|displaystyle|textstyle)\b", " ", str(text or ""))
    value = re.sub(r"\\(?:begin|end)\{[^{}]*\}", "", value)
    return re.sub(r"\s+", " ", value).strip()


Token = tuple[str, str]


def _tokenize(value: str) -> list[Token]:
    pattern = re.compile(r"\s*(?:(\d+(?:[.,]\d+)?)|([A-Za-zÁÉÍÓÚÜÑáéíóúüñπθΔΣ]+)|(≤|≥|≠|<=|>=|!=|[+\-*/=^_(),·×]))")
    tokens: list[Token] = []
    position = 0
    while position < len(value):
        if value[position:].strip() == "":
            break
        match = pattern.match(value, position)
        if not match:
            if value[position] in "[]{}":
                position += 1
                continue
            raise ValueError(f"unknown_math_token:{value[position:position + 8]}")
        number, identifier, operator = match.groups()
        if number:
            tokens.append(("number", number))
        elif identifier:
            if identifier.isascii() and identifier.isalpha() and len(identifier) > 1 and identifier.casefold() not in _KNOWN_IDENTIFIERS:
                tokens.extend(("identifier", character) for character in identifier)
            else:
                tokens.append(("identifier", identifier))
        else:
            tokens.append(("operator", operator))
        position = match.end()
    return _insert_implicit_products(tokens)


def _insert_implicit_products(tokens: list[Token]) -> list[Token]:
    if not tokens:
        return []
    output: list[Token] = []
    for token in tokens:
        if output and _token_ends_primary(output[-1]) and _token_starts_primary(token):
            function_call = output[-1][0] == "identifier" and token == ("operator", "(")
            prefix_operator = output[-1] in {("identifier", "sum"), ("identifier", "integral")}
            if not function_call and not prefix_operator:
                output.append(("operator", "*"))
        output.append(token)
    return output


def _token_ends_primary(token: Token) -> bool:
    return token[0] in {"number", "identifier"} or token == ("operator", ")")


def _token_starts_primary(token: Token) -> bool:
    return token[0] in {"number", "identifier"} or token == ("operator", "(")


class _FormulaParser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.position = 0

    def parse(self, minimum: int = 0):
        left = self._primary()
        precedence = {"=": 1, "<=": 1, ">=": 1, "!=": 1, "≤": 1, "≥": 1, "≠": 1, "+": 2, "-": 2, "*": 3, "/": 3, "·": 3, "×": 3}
        while self.position < len(self.tokens):
            operator = self.tokens[self.position][1]
            level = precedence.get(operator, -1)
            if level < minimum:
                break
            self.position += 1
            right = self.parse(level + 1)
            left = ("binary", operator, left, right)
        return left

    def _primary(self):
        if self.position >= len(self.tokens):
            raise ValueError("missing_formula_operand")
        kind, value = self.tokens[self.position]
        self.position += 1
        if value == "-":
            node = ("unary", self._primary())
        elif value == "(":
            node = self.parse()
            self._expect(")")
            node = ("group", node)
        elif kind in {"number", "identifier"}:
            node = ("atom", value)
            if kind == "identifier" and self._peek("("):
                self.position += 1
                arguments = []
                if not self._peek(")"):
                    while True:
                        arguments.append(self.parse())
                        if not self._peek(","):
                            break
                        self.position += 1
                self._expect(")")
                node = ("call", value, arguments)
        else:
            raise ValueError("invalid_formula_primary")
        while self.position < len(self.tokens) and self.tokens[self.position][1] in {"^", "_"}:
            operation = self.tokens[self.position][1]
            self.position += 1
            argument = self._primary()
            node = ("power" if operation == "^" else "subscript", node, argument)
        return node

    def _peek(self, value: str) -> bool:
        return self.position < len(self.tokens) and self.tokens[self.position][1] == value

    def _expect(self, value: str) -> None:
        if not self._peek(value):
            raise ValueError(f"expected_{value}")
        self.position += 1


def _speak_node(node) -> str:
    kind = node[0]
    if kind == "atom":
        return _speak_atom(node[1])
    if kind == "group":
        return _speak_node(node[1])
    if kind == "unary":
        return "menos " + _speak_node(node[1])
    if kind == "binary":
        words = {"+": "más", "-": "menos", "*": "por", "·": "producto punto", "×": "producto cruz", "/": "dividido para", "=": "es igual a", "<=": "menor o igual que", "≤": "menor o igual que", ">=": "mayor o igual que", "≥": "mayor o igual que", "!=": "distinto de", "≠": "distinto de"}
        return f"{_speak_node(node[2])} {words[node[1]]} {_speak_node(node[3])}"
    if kind == "power":
        base = _speak_node(node[1])
        exponent = _speak_node(node[2])
        grouped = node[1][0] == "group"
        suffix = "todo al cuadrado" if exponent == "dos" and grouped else "al cuadrado" if exponent == "dos" else "todo al cubo" if exponent == "tres" and grouped else "al cubo" if exponent == "tres" else f"elevado a {exponent}"
        return f"{base}, {suffix}"
    if kind == "subscript":
        return f"{_speak_node(node[1])} sub {_speak_node(node[2])}"
    if kind == "call":
        name = node[1].casefold()
        args = [_speak_node(item) for item in node[2]]
        if name == "sqrt":
            return "raíz cuadrada de " + (args[0] if args else "la expresión")
        if name == "frac":
            return f"{args[0]} dividido para {args[1]}" if len(args) >= 2 else "fracción"
        if name in {"vec", "vector"}:
            return "vector " + (args[0] if args else "")
        return f"{node[1]} de " + ", ".join(args)
    raise ValueError("unknown_formula_node")


def _speak_atom(value: str) -> str:
    numbers = {"0": "cero", "1": "uno", "2": "dos", "3": "tres"}
    operators = {"dot": "producto punto", "cross": "producto cruz"}
    if value.casefold() in operators:
        return operators[value.casefold()]
    return numbers.get(value, _GREEK.get(value, value))


def _token_fallback(value: str) -> str:
    clean = safe_speech_fallback(value)
    clean = re.sub(r"\bdot\b", "producto punto", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bcross\b", "producto cruz", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsqrt\b", "raíz cuadrada de", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bsum\b", "sumatoria", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\bintegral\b", "integral", clean, flags=re.IGNORECASE)
    return clean
