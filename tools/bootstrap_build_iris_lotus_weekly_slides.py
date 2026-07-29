"""Install the small PPTX dependency if absent, then build the weekly deck."""

from __future__ import annotations

import importlib.util
import subprocess
import sys


if importlib.util.find_spec("pptx") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-pptx"])

from build_iris_lotus_weekly_slides import build_deck


build_deck()
