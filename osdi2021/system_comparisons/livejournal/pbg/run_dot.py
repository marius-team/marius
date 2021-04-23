import argparse
import random
from pathlib import Path

import attr
import pkg_resources
from torchbiggraph.config import ConfigFileLoader, add_to_sys_path
from torchbiggraph.converters.importers import TSVEdgelistReader, convert_input_data
from torchbiggraph.converters.utils import download_url, extract_gzip
from torchbiggraph.eval import do_eval
from torchbiggraph.train import train
from torchbiggraph.util import (
    SubprocessInitializer,
    set_logging_verbosity,
    setup_logging,
)

import sys
from os import path
import os

sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))))
from osdi2021.utils import make_tsv

URL = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
TRAIN_FILENAME = "lj_train.txt"
VALID_FILENAME = "lj_valid.txt"
TEST_FILENAME = "lj_valid.txt"
FILENAMES = [TRAIN_FILENAME, VALID_FILENAME, TEST_FILENAME]

TRAIN_FRACTION = .9
VALID_FRACTION = .05
TEST_FRACTION = .05

def random_split_file(fpath: Path) -> None:
    train_file = fpath.parent / TRAIN_FILENAME
    valid_file = fpath.parent / VALID_FILENAME
    test_file = fpath.parent / TEST_FILENAME

    if train_file.exists() and test_file.exists():
        print(
            "Found some files that indicate that the input data "
            "has already been shuffled and split, not doing it again."
        )
        print(f"These files are: {train_file} and {test_file}")
        return

    print("Shuffling and splitting train/test file. This may take a while.")

    print(f"Reading data from file: {fpath}")
    with fpath.open("rt") as in_tf:
        lines = in_tf.readlines()

    # The first few lines are comments
    lines = lines[4:]
    print("Shuffling data")
    random.shuffle(lines)
    split_len = int(len(lines) * TRAIN_FRACTION)

    valid_len = int(len(lines) * (TRAIN_FRACTION + VALID_FRACTION))

    print("Splitting to train and test files")
    with train_file.open("wt") as out_tf_train:
        for line in lines[:split_len]:
            out_tf_train.write(line)

    with valid_file.open("wt") as out_tf_train:
        for line in lines[split_len:valid_len]:
            out_tf_train.write(line)

    with test_file.open("wt") as out_tf_test:
        for line in lines[valid_len:]:
            out_tf_test.write(line)

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Example on Livejournal")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument(
        "--data_dir", type=Path, default="data", help="where to save processed data"
    )

    args = parser.parse_args()

    # download data
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    fpath = download_url(URL, data_dir)
    fpath = extract_gzip(fpath)
    print("Downloaded and extracted file.")

    # random split file for train and test
    random_split_file(fpath)

    loader = ConfigFileLoader()
    config = loader.load_config(args.config, args.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    input_edge_paths = [data_dir / name for name in FILENAMES]
    output_train_path, output_valid_path, output_test_path = config.edge_paths

    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=1, rel_col=None),
        dynamic_relations=config.dynamic_relations,
    )

    train_config = attr.evolve(config, edge_paths=[output_train_path])
    train(train_config, subprocess_init=subprocess_init)

    eval_config = attr.evolve(config, edge_paths=[output_test_path])
    make_tsv(eval_config, False)

    os.makedirs("pbg_embeddings/live_journal/edges/train", exist_ok=True)
    os.makedirs("pbg_embeddings/live_journal/edges/evaluation", exist_ok=True)
    os.makedirs("pbg_embeddings/live_journal/edges/test", exist_ok=True)
    os.makedirs("pbg_embeddings/live_journal/embeddings", exist_ok=True)
    os.makedirs("pbg_embeddings/live_journal/relations", exist_ok=True)

    os.system("touch pbg_embeddings/live_journal/edges/train/edges.bin")
    os.system("touch pbg_embeddings/live_journal/edges/evaluation/edges.bin")
    os.system("mv edges.bin pbg_embeddings/live_journal/edges/test/")

    os.system("mv embeddings.bin pbg_embeddings/live_journal/embeddings/")
    os.system("mv lhs_relations.bin pbg_embeddings/live_journal/relations/")
    os.system("mv rhs_relations.bin pbg_embeddings/live_journal/relations/")

    print("Exported embeddings")

    os.system("marius_eval osdi2021/system_comparisons/livejournal/pbg/dot_eval.ini --path.train_edges foo")


if __name__ == "__main__":
    main()