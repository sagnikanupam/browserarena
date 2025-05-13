"""
Logging helpers (anonymisation, masking, …) for BrowserArena.
"""
from __future__ import annotations

import logging
from typing import Dict


class SensitiveDataFilter(logging.Filter):
    """
    Replace every occurrence of user-supplied *sensitive values* inside a
    log record with the token ``<secret>{placeholder}</secret>`` before the
    record is emitted.

    Parameters
    ----------
    sensitive_data :
        Dict that maps *placeholder* ➜ *real value*.
        Example: ``{"USER_ID": "john_doe_42"}``
    """

    def __init__(self, sensitive_data: Dict[str, str] | None = None) -> None:
        super().__init__()
        self.sensitive_data: Dict[str, str] = sensitive_data or {}

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #
    def _mask(self, text: str) -> str:
        for placeholder, real_value in self.sensitive_data.items():
            if real_value:
                text = text.replace(real_value, f"<secret>{placeholder}</secret>")
        return text

    # ------------------------------------------------------------------ #
    # logging.Filter hook
    # ------------------------------------------------------------------ #
    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        """Mutate *record* in place; always return ``True`` to keep the record."""
        record.msg = self._mask(str(record.msg))

        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(self._mask(str(a)) for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: self._mask(str(v)) for k, v in record.args.items()}
            else:
                record.args = self._mask(str(record.args))

        return True 