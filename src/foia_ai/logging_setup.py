import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Set up basic logging to stdout."""
    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, handlers=[handler], format=fmt)
