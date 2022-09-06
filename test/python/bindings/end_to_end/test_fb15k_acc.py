import os
import shutil
import unittest
from pathlib import Path
from test.python.constants import TMP_TEST_DIR

import pytest


class TestFB15K(unittest.TestCase):
    @classmethod
    def setUp(self):
        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

    @classmethod
    def tearDown(self):
        pass
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    @pytest.mark.skipif(os.environ.get("MARIUS_NO_BINDINGS", None) == "TRUE", reason="Requires building the bindings")
    def test_one_epoch(self):
        pass
