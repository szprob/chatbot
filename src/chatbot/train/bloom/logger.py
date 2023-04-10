import logging
import os


def get_logger(logger_name, output_dir):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    os.makedirs(output_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"), mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="""%(asctime)s - %(filename)s[line:%(lineno)d]
            - %(levelname)s: %(message)s""",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter(
            fmt="""%(asctime)s - %(filename)s[line:%(lineno)d]
            - %(levelname)s: %(message)s""",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(console_handler)
    return logger
