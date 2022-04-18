import unittest
import shutil
import os
from pathlib import Path

import pandas as pd
from marius.tools.marius_predict import run_predict, set_args
from test.test_data.generate import generate_random_dataset
from test.test_configs.generate_test_configs import generate_configs_for_dataset
from test.python.constants import TMP_TEST_DIR
import marius as m


def validate_metrics(config, metrics, num_items, output_dir=None):

    if output_dir is None:
        metrics_file = Path(config.storage.dataset.base_directory) / Path("metrics.txt")

    else:
        metrics_file = Path(output_dir) / Path("metrics.txt")

    assert metrics_file.exists()

    if config.model.learning_task == m.config.LearningTask.LINK_PREDICTION:
        task = "Link Prediction:"

        if config.model.decoder.options.inverse_edges:
            factor = 2
        else:
            factor = 1
    else:
        task = "Node Classification:"
        factor = 1

    found = []
    for _ in metrics:
        found.append(False)

    report_items = -1
    with open(metrics_file) as f:
        for line in f.readlines():
            if line.startswith(task):
                report_items = int(line.split(": ")[-1].split(" ")[0])
            else:
                for i, metric in enumerate(metrics):
                    if line.upper().startswith(metric.upper()):
                        found[i] = True

    # Check that all metrics have been found in the report
    for f in found:
        assert f

    # Check the report contains the correct amount of evaluation edges/nodes
    assert report_items == factor * num_items


def validate_scores(config, num_edges, save_scores, save_ranks, output_dir=None):

    if output_dir is None:
        scores_file = Path(config.storage.dataset.base_directory) / Path("scores.csv")
    else:
        scores_file = Path(output_dir) / Path("scores.csv")

    assert scores_file.exists()

    scores_df = pd.read_csv(scores_file, delimiter="", header=None)

    if config.storage.dataset.num_relations > 1:
        num_cols = 3
    else:
        num_cols = 2

    if save_scores:
        num_cols += 1

    if save_ranks:
        num_cols += 1

    assert scores_df.shape[0] == num_edges
    assert scores_df.shape[1] == num_cols


def validate_labels(config, num_nodes, output_dir=None):

    if output_dir is None:
        labels_file = Path(config.storage.dataset.base_directory) / Path("labels.csv")
    else:
        labels_file = Path(output_dir) / Path("labels.csv")

    assert labels_file.exists()

    labels_df = pd.read_csv(labels_file, delimiter="", header=None)
    num_cols = 2

    assert labels_df.shape[0] == num_nodes
    assert labels_df.shape[1] == num_cols


class TestPredictLP(unittest.TestCase):
    config_file = None

    @classmethod
    def setUp(self):

        if not Path(TMP_TEST_DIR).exists():
            Path(TMP_TEST_DIR).mkdir()

        base_dir = TMP_TEST_DIR

        num_nodes = 100
        num_rels = 10
        num_edges = 1000

        name = "basic_lp"
        generate_random_dataset(output_dir=base_dir / Path(name),
                                num_nodes=num_nodes,
                                num_edges=num_edges,
                                num_rels=num_rels,
                                splits=[.9, .05, .05],
                                task="lp")

        generate_configs_for_dataset(base_dir / Path(name),
                                     model_names=["distmult"],
                                     storage_names=["in_memory"],
                                     training_names=["sync"],
                                     evaluation_names=["sync"],
                                     task="lp")

        for filename in os.listdir(base_dir / Path(name)):
            if filename.startswith("M-"):
                self.config_file = base_dir / Path(name) / Path(filename)

        config = m.config.loadConfig(self.config_file.__str__(), True)
        self.config_file = Path(config.storage.model_dir) / Path("full_config.yaml")
        m.manager.marius_train(config)

    @classmethod
    def tearDown(self):
        if Path(TMP_TEST_DIR).exists():
            shutil.rmtree(Path(TMP_TEST_DIR))

    def test_basic_lp(self):
        parser = set_args()
        args = parser.parse_args(["--config", self.config_file.__str__(), "--metrics", "mrr"])
        run_predict(args)

        config = m.config.loadConfig(self.config_file.__str__(), save=False)
        validate_metrics(config, ["MRR"], config.storage.dataset.num_test)

    def test_lp_metrics(self):
        parser = set_args()
        args = parser.parse_args(["--config", self.config_file.__str__(), "--metrics", "mrr", "mr", "hits1", "hits2", "hits3", "hits4", "hits5", "hits10", "hits20"])
        run_predict(args)

        config = m.config.loadConfig(self.config_file.__str__(), save=False)
        validate_metrics(config, ["MRR", "MEAN RANK", "HITS@1", "HITS@2", "HITS@3", "HITS@4", "HITS@5", "HITS@10", "HITS@20"], config.storage.dataset.num_test)

    def test_lp_save_ranks(self):
        pass

    def test_lp_save_scores(self):
        pass
