import logging
import os


def setup_logging(log_file: str, resume: bool = False, dummy: bool = False):
    """Setup logging configuration."""
    if dummy:
        logging.getLogger('dummy')
        return

    file_mode = 'a' if os.path.isfile(log_file) and resume else 'w'

    root_logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    # Remove all existing handlers (can't use the `force` option with
    # python < 3.8)
    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)
    # Add the handlers we want to use
    fileout = logging.FileHandler(log_file, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(fileout)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console)
