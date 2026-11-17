"""Test-only path mapping for the official runner on a host without Docker."""

import os
from pathlib import Path
import sys

from contest_harness.cli import main
from contest_harness import predict, round


original = round.ParentView.__init__


def parent_for_test(self, frame):
    original(self, frame)
    self.root = Path(os.environ["ZHIXUN_TEST_PREVIOUS"])


round.ParentView.__init__ = parent_for_test
predict.model_dir = lambda: Path(os.environ["ZHIXUN_TEST_MODEL"])
raise SystemExit(main(sys.argv))
