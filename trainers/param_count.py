"""Torch-free parsing of parameter counts such as ``100k`` or ``1.5M``.

This module must stay pure Python (no torch import): the desktop GUI imports
it to validate the "Trainable parameter budget" field, and the GUI process is
never allowed to import torch. ``trainers/param_budget.py`` re-exports
``parse_param_count`` for the training code and ``cli.py``.
"""
from __future__ import annotations

import re
from argparse import ArgumentTypeError
from decimal import Decimal, InvalidOperation


class ParamCountError(ArgumentTypeError, ValueError):
    """Invalid parameter count. An ``ArgumentTypeError`` so argparse prints the
    message verbatim, and a ``ValueError`` so plain Python callers can catch it."""


_SUFFIXES = {"": 1, "k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "g": 1_000_000_000}
_PATTERN = re.compile(r"^([0-9]*\.?[0-9]+)\s*([kmbg]?)$", re.IGNORECASE)
# Commas are only accepted as thousands separators ("1,000,000"), so a
# decimal comma such as "1,5M" is rejected instead of silently becoming 15M.
_THOUSANDS = re.compile(r"^[0-9]{1,3}(,[0-9]{3})+(\.[0-9]+)?\s*[kmbg]?$", re.IGNORECASE)


def parse_param_count(text) -> int:
    """Parse a parameter count like ``250000``, ``100k``, ``1.5M`` or ``2m``.

    Underscores, commas and surrounding spaces are ignored (``1_000_000``,
    ``1,000,000``). Suffixes: ``k`` = thousand, ``m`` = million, ``b``/``g`` =
    billion (case-insensitive). The result must be a whole number of
    parameters greater than zero.

    Raises ``ParamCountError`` (both an ``argparse.ArgumentTypeError`` and a
    ``ValueError``) on junk, zero, negative or fractional counts, so it can be
    used directly as an argparse ``type=`` and argparse shows the message.
    """
    if isinstance(text, bool):
        raise ParamCountError(f"invalid parameter count: {text!r}")
    if isinstance(text, int):
        if text <= 0:
            raise ParamCountError(f"parameter count must be greater than zero, got {text}")
        return text
    raw = str(text)
    cleaned = raw.strip().replace("_", "")
    if "," in cleaned and _THOUSANDS.match(cleaned):
        cleaned = cleaned.replace(",", "")
    match = _PATTERN.match(cleaned)
    if not match:
        raise ParamCountError(
            f"invalid parameter count {raw!r}: use a whole number or a number with a "
            f"k/M suffix, e.g. 250000, 100k or 1.5M"
        )
    number, suffix = match.groups()
    try:
        value = Decimal(number) * _SUFFIXES[suffix.lower()]
    except InvalidOperation as exc:  # pragma: no cover - the regex already guards this
        raise ParamCountError(f"invalid parameter count {raw!r}") from exc
    if value != value.to_integral_value():
        raise ParamCountError(f"parameter count {raw!r} is not a whole number of parameters")
    value = int(value)
    if value <= 0:
        raise ParamCountError(f"parameter count must be greater than zero, got {raw!r}")
    return value


def param_count_issue(text: str) -> str | None:
    """The error message for ``text``, or None when it is blank or valid.

    Blank means "no budget" in the GUI, so it is never an issue here.
    """
    if text is None or not str(text).strip():
        return None
    try:
        parse_param_count(text)
    except ParamCountError as exc:
        return str(exc)
    return None
