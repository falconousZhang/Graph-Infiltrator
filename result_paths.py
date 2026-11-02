"""Shared paths for keeping generated results outside the source tree."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = PROJECT_ROOT.parent / f'{PROJECT_ROOT.name}_results'
RESULTS_ROOT = Path(
    os.environ.get('GRAPH_INFILTRATOR_RESULTS_ROOT', str(DEFAULT_RESULTS_ROOT))
).expanduser().resolve()


def results_path(*parts):
    """Return a path below the configurable external results root."""

    return RESULTS_ROOT.joinpath(*parts)
