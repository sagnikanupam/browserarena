"""
Helpers for log handling & sanitisation.
"""
from __future__ import annotations

import logging
from typing import Dict


class SensitiveDataFilter(logging.Filter):
    """
    A logging filter that replaces all occurrences of *real* sensitive values
    with the placeholder ``<secret>{placeholder}</secret>`` before the record
    is emitted.  
    The mapping is taken from the ``sensitive_data`` dictionary that can be
    supplied when instantiating an `Agent`.
    """

    def __init__(self, sensitive_data: Dict[str, str] | None = None) -> None:
        super().__init__()
        # fallback to empty dict to avoid ``None`` checks later on
        self.sensitive_data: Dict[str, str] = sensitive_data or {}

    # --------------------------------------------------------------------- #
    # private helpers
    # --------------------------------------------------------------------- #
    def _mask(self, value: str) -> str:
        # Convert any non-string input first
        if not isinstance(value, str):
            value = str(value)

        for placeholder, real_value in self.sensitive_data.items():
            if real_value:
                value = value.replace(real_value, f"<secret>{placeholder}</secret>")
        return value

    # --------------------------------------------------------------------- #
    # logging.Filter hook
    # --------------------------------------------------------------------- #
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        """
        Mutates the record **in-place** – masking the message and all args.
        Always returns *True* so that the record is still propagated.
        """
        record.msg = self._mask(record.msg)

        # ``record.args`` can be a dict / tuple / str – handle the common cases
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(self._mask(arg) for arg in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self._mask(v) for k, v in record.args.items()}
            else:
                record.args = self._mask(str(record.args))
        return True 