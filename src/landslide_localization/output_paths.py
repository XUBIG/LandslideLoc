"""Keep generated files inside a dedicated output directory."""
import os
from pathlib import Path
from landslide_localization.runtime_paths import REPO_ROOT

EXTERNAL_OUTPUT_ROOT_ENV = "LANDSLIDE_OUTPUT_ROOT"

def ensure_output_path(value):
    root = Path(os.environ.get(EXTERNAL_OUTPUT_ROOT_ENV, REPO_ROOT / "outputs")).resolve()
    if root == Path(root.anchor):
        raise ValueError("Choose a dedicated output directory")
    path = Path(value).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"Output must be inside {root}: {path}")
    return path
