
from pathlib import Path

import pytest


# so can work w/ pytest called from repo root, but also w/ scripts like
# generate_reference_outputs_for_repro.py, which I've been calling from this directory.
test_dir: Path = Path(__file__).resolve().parent

# TODO define in conftest or something?
# TODO refactor this handling of test data path? also used in test_al_analysis.py
test_data_dir: Path = test_dir / 'test_data'

