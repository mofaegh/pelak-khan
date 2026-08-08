from __future__ import annotations


# Baseline v1: standard Iranian private-vehicle plate as OCR text:
# DD L DDD DD -> DDLDDDDD (8 characters)
def validate_standard_plate(
    text: str,
    allowed_letters: set[str],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if len(text) != 8:
        return False, [f"invalid_length_{len(text)}"]

    if not text[:2].isdigit():
        reasons.append("first_two_not_digits")

    if text[2] not in allowed_letters:
        reasons.append("position_3_not_valid_letter")

    if not text[3:].isdigit():
        reasons.append("last_five_not_digits")

    return not reasons, reasons


def display_plate(text: str) -> str:
    if len(text) == 8:
        return f"{text[:2]} {text[2]} {text[3:6]} | {text[6:]}"
    return text


def edge_suspect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    margin_frac: float,
) -> bool:
    mx = width * margin_frac
    my = height * margin_frac
    return (
        x1 <= mx
        or y1 <= my
        or x2 >= (width - mx)
        or y2 >= (height - my)
    )


def expand_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
    pad_frac: float,
) -> tuple[int, int, int, int]:
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * pad_frac
    pad_y = box_height * pad_frac
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )
