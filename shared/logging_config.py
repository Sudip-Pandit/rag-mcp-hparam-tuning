"""Centralized logging with optional PII redaction (Part 2.5 guardrails)."""

import logging
import re
import sys

_EMAIL = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE = re.compile(r"\b\+?\d[\d\-\s().]{7,}\d\b")


class PIIRedactingFilter(logging.Filter):
    """Strips emails, SSNs and phone numbers from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        msg = _EMAIL.sub("[EMAIL]", msg)
        msg = _SSN.sub("[SSN]", msg)
        msg = _PHONE.sub("[PHONE]", msg)
        record.msg = msg
        record.args = ()
        return True


def configure_logging(level: int = logging.INFO, redact_pii: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    if redact_pii:
        handler.addFilter(PIIRedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


if __name__ == "__main__":
    configure_logging()
    log = logging.getLogger("demo")
    log.info("User jane.doe@example.com with SSN 123-45-6789 called a tool")
