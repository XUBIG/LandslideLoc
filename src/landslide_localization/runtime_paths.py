"""Portable project paths shared by command-line and subprocess workers."""
from pathlib import Path
import os

REPO_ROOT = Path(os.environ.get("LANDSLIDE_REPO_ROOT", Path(__file__).resolve().parents[2])).resolve()


def configure_numerical_threads():
    """Use one numerical-library thread per worker unless explicitly configured."""
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
