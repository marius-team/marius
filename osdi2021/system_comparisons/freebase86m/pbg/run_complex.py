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

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Example on Livejournal")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("-p", "--param", action="append", nargs="*")
    parser.add_argument(
        "--data_dir", type=Path, default="data", help="where to save processed data"
    )

    args = parser.parse_args()

    loader = ConfigFileLoader()
    config = loader.load_config(args.config, args.param)
    set_logging_verbosity(config.verbose)
    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)
    output_train_path, output_valid_path, output_test_path = config.edge_paths

    train_config = attr.evolve(config, edge_paths=[output_train_path])
    train(train_config, subprocess_init=subprocess_init)

    eval_config = attr.evolve(config, edge_paths=[output_test_path])
    make_tsv(eval_config, True)

    os.makedirs("pbg_embeddings/freebase86m_16/edges/train", exist_ok=True)
    os.makedirs("pbg_embeddings/freebase86m_16/edges/evaluation", exist_ok=True)
    os.makedirs("pbg_embeddings/freebase86m_16/edges/test", exist_ok=True)
    os.makedirs("pbg_embeddings/freebase86m_16/embeddings", exist_ok=True)
    os.makedirs("pbg_embeddings/freebase86m_16/relations", exist_ok=True)

    os.system("touch pbg_embeddings/freebase86m_16/edges/train/edges.bin")
    os.system("touch pbg_embeddings/freebase86m_16/edges/evaluation/edges.bin")
    os.system("mv edges.bin pbg_embeddings/freebase86m_16/edges/test/")

    os.system("mv embeddings.bin pbg_embeddings/freebase86m_16/embeddings/")
    os.system("mv lhs_relations.bin pbg_embeddings/freebase86m_16/relations/")
    os.system("mv rhs_relations.bin pbg_embeddings/freebase86m_16/relations/")

    print("Exported embeddings")

    os.system("marius_eval osdi2021/system_comparisons/freebase86m/pbg/complex_eval.ini --path.train_edges foo")


if __name__ == "__main__":
    main()