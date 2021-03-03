import logging
import sys

def init_logging(capture_warnings = False):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s;%(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logging.captureWarnings(capture_warnings)