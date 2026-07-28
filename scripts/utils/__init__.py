"""Shared paths and logging setup.

Deliberately thin. This package used to eagerly re-export a cross-validation /
memory-monitor / model-utils layer from the TensorFlow era; all three fail to
import in the current runtime (`cross_validation` reaches for
`scripts.train_models`, deleted in the Phase 1 cleanup; the other two want
`psutil`, which is not in environment.yml). The failures were swallowed, but
each one logged a WARNING, so every production run - post-draw routine, EV
email alert, cron - opened with four alarming lines about modules nothing
imports. Live code needs exactly two things from here: LOG_DIR and
setup_logging.
"""

import logging
from pathlib import Path

__version__ = '0.2.0'

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
LOG_DIR = ROOT_DIR / "logs"

LOG_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True, parents=True)

logger = logging.getLogger(__name__)


# Defined BEFORE any intra-package imports: modules imported from this __init__
# (e.g. validations.data_validator) import setup_logging back from this package
# mid-initialization, and it must already exist by then - otherwise Python
# silently resolves the name to the setup_logging SUBMODULE.
def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOG_DIR / 'lottery.log'),
            logging.StreamHandler()
        ]
    )


__all__ = [
    'setup_logging',
    'ROOT_DIR',
    'DATA_DIR',
    'MODEL_DIR',
    'OUTPUT_DIR',
    'LOG_DIR',
]
