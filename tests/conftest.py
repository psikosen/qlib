import os
import sys
from pathlib import Path

"""Ignore RL tests on non-linux platform."""
collect_ignore = []

# Ensure the repository root is importable so ``import qlib`` works when tests
# are executed from within the ``tests`` directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform != "linux":
    for root, dirs, files in os.walk("rl"):
        for file in files:
            collect_ignore.append(os.path.join(root, file))
