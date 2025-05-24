import logging
import os


def setup_logger(logs_name: str):
    logger = logging.getLogger(f"{logs_name}_logger")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        os.makedirs("logs", exist_ok=True)
        log_file = os.path.join("logs", f"{logs_name}.log")
        main_log_file = os.path.join("logs", "all.log")

        file_handler = logging.FileHandler(log_file, mode="w")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%H:%M:%S")
        file_handler.setFormatter(formatter)

        main_log_handler = logging.FileHandler(main_log_file, mode="a")
        main_log_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(main_log_handler)

    return logger
