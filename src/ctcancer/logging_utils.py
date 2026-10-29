"""Logging that never writes patient identifiers.

`PHIFilter` redacts values that look like DICOM PatientName / PatientID / dates of birth
if they slip into a log message.
"""
from __future__ import annotations

import logging
import re

_PHI_PATTERNS = [
    re.compile(r"(PatientName\s*[=:]\s*)\S+", re.I),
    re.compile(r"(PatientID\s*[=:]\s*)\S+", re.I),
    re.compile(r"(PatientBirthDate\s*[=:]\s*)\S+", re.I),
]


class PHIFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pat in _PHI_PATTERNS:
            msg = pat.sub(r"\1[REDACTED]", msg)
        record.msg, record.args = msg, ()
        return True


def get_logger(name: str = "ctcancer", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        h.addFilter(PHIFilter())
        logger.addHandler(h)
        logger.setLevel(level)
        logger.propagate = False
    return logger
