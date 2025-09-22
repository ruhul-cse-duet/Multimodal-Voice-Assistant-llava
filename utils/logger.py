"""
Logging utilities for the Multimodal Voice Assistant
"""
import logging
import datetime
from pathlib import Path
from config.settings import LOGGING_CONFIG

def setup_logger(name: str = "multimodal_assistant") -> logging.Logger:
    """
    Set up logger with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_CONFIG["log_level"]))
    
    # Avoid adding multiple handlers if logger already exists
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(LOGGING_CONFIG["log_format"])
    
    # File handler
    file_handler = logging.FileHandler(LOGGING_CONFIG["log_file"])
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def write_history(text: str, log_file: str = None) -> None:
    """
    Write text to history log file
    
    Args:
        text: Text to write
        log_file: Optional custom log file path
    """
    if log_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGGING_CONFIG["log_file"].parent / f"{timestamp}_history.txt"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"{datetime.datetime.now().isoformat()}: {text}\n")
