import unittest
import subprocess
import shutil
from pathlib import Path
import pytest
import os
import marius as m
from marius.tools import preprocess
from marius.tools.config_generator import set_up_files


class TestFB15K(unittest.TestCase):
    output_path = None

    @classmethod
    def setUp(self):
        self.output_path = set_up_files("output_dir")

    @classmethod
    def tearDown(self):
        if Path(self.output_path).exists():
            shutil.rmtree(Path(self.output_path))
        if Path("training_data").exists():
            shutil.rmtree(Path("training_data"))

    @pytest.mark.skipif(os.environ.get("MARIUS_ONLY_PYTHON", None) == "TRUE", reason="Requires building the bindings")
    def test_one_epoch(self):
        preprocess.fb15k(output_dir="output_dir/")
        config_path = "examples/training/configs/fb15k_cpu.ini"
        config = m.parseConfig(config_path)

        train_set, eval_set = m.initializeDatasets(config)

        model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

        trainer = m.SynchronousTrainer(train_set, model)
        evaluator = m.SynchronousEvaluator(eval_set, model)

        trainer.train(1)
        evaluator.evaluate(True)
