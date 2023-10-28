import os
import unittest
from pathlib import Path

from marius.tools.mpic.compiler import run_compiler

examples_dir = Path(Path(__file__).parent, "examples").resolve()
TEST_GEN_DIR = Path(os.getcwd(), "build/mpic_gen").resolve()


def run_codegen_test(filename):
    run_compiler(os.path.join(examples_dir, filename))
    return os.path.exists(os.path.join(TEST_GEN_DIR, Path(filename).with_suffix(".h"))) and os.path.exists(
        os.path.join(TEST_GEN_DIR, Path(filename).with_suffix(".cpp"))
    )


class TestCodeGen(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(run_codegen_test("basic_layer.py"))
