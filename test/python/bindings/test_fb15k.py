import unittest
import subprocess
import shutil
from pathlib import Path
import pytest
import os
import marius as m
from marius.tools import preprocess


class TestFB15K(unittest.TestCase):

    @classmethod
    def tearDown(self):
        if Path("output_dir").exists():
            shutil.rmtree(Path("output_dir"))
        if Path("training_data").exists():
            shutil.rmtree(Path("training_data"))

    @pytest.mark.skipif(os.environ.get("MARIUS_ONLY_PYTHON", None) == "TRUE", reason="Requires building the bindings")
    def test_one_epoch(self):
        preprocess.fb15k("output_dir/", "output_dir/")
        config_path = "examples/training/configs/fb15k_cpu.ini"
        config = m.parseConfig(config_path)

        train_set, eval_set = m.initializeDatasets(config)

        model = m.initializeModel(config.model.encoder_model, config.model.decoder_model)

        trainer = m.SynchronousTrainer(train_set, model)
        evaluator = m.SynchronousEvaluator(eval_set, model)

        trainer.train(1)
        evaluator.evaluate(True)
