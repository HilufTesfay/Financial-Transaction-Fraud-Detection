import logging
import pathlib as path


def get_root_path() -> path:
    """returns the root path of the project"""
    return path.Path(__file__).resolve().parent.parent.parent


def setup_logger(file: str):
    """setting up global logger"""
    log_file = get_root_path() / "logs" / file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{log_file}.log", mode="a", encoding="utf-8"),
        ],
    )
