# src/logging_config.py
import logging
import re
import sys

from pythonjsonlogger import jsonlogger


class RequestIDFilter(logging.Filter):
    def filter(self, record):
        record.request_id = getattr(record, "request_id", "N/A")
        return True


class RedactingFilter(logging.Filter):
    def filter(self, record):
        secret_patterns = [
            r"API_KEY=[^,\s]+",
            r"token\s*[:=]\s*[^\s]+",
            r"sk-[A-Za-z0-9]{32,}",
        ]
        for pattern in secret_patterns:
            record.msg = re.sub(pattern, "***REDACTED***", record.msg)
        return True


def setup_structured_logging(log_level=logging.INFO):
    logger = logging.getLogger()
    if logger.handlers:
        return

    logger.setLevel(log_level)
    log_handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(funcName)s %(lineno)d %(request_id)s %(message)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "name": "logger_name",
            "funcName": "function",
            "lineno": "line_number",
        },
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    log_handler.setFormatter(formatter)
    log_handler.addFilter(RequestIDFilter())
    log_handler.addFilter(RedactingFilter())
    logger.addHandler(log_handler)

    logging.getLogger("google").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("streamlit").setLevel(logging.WARNING)

    logger.info("Structured JSON logging configured successfully.")
