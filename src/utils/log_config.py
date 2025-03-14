import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "file": record.filename,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_obj)

def setup_logger(
        log_file_path: str = "log/app.log",
        max_file_size: int = 10 * 1024 * 1024,
        backup_count: int = 10,
        log_level: int = logging.DEBUG
    ) -> None:
    
    # Get absolute path
    log_path = Path(log_file_path).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger("root")
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()  # Remove any existing handlers

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_file_size,
        backupCount=backup_count,
        mode='a'
    )
    json_formatter = JSONFormatter()
    file_handler.setFormatter(json_formatter)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)
    
    # Test logging
    root_logger.info("Logging system initialized")

def get_logger(name: str = None) -> logging.Logger:
    return logging.getLogger(name or __name__)