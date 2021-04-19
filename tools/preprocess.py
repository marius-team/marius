import argparse
import gzip
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch

from config_generator import output_config
from config_generator import output_bash_cmds
from csv_converter import general_parser


def live_journal(output_dir, num_partitions=1, split=(.05, .05)):
    download_path = download_file("https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz", output_dir)
    extract_file(download_path)
    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("soc-LiveJournal1.txt"))], ["sd"],
                                                 [output_dir], num_partitions=num_partitions, dataset_split=split)
    output_config(stats, num_nodes, num_edges, output_dir, "live_journal")
    output_config(stats, num_nodes, num_edges, output_dir, "live_journal", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "live_journal", device = "multi_gpu")


def fb15k(output_dir, num_partitions=1):
    download_path = download_file("https://dl.fbaipublicfiles.com/starspace/fb15k.tgz", output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("FB15k")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("FB15k")).rmdir()

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("freebase_mtr100_mte100-train.txt")),
                                                  str(Path(output_dir) / Path("freebase_mtr100_mte100-valid.txt")),
                                                  str(Path(output_dir) / Path("freebase_mtr100_mte100-test.txt"))],
                                                 ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k")
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k", device = "multi_gpu")


def twitter(output_dir, num_partitions=1, split=(.05, .05)):
    download_path = download_file("https://snap.stanford.edu/data/twitter-2010.txt.gz", output_dir)
    extract_file(download_path)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("twitter-2010.txt"))], ["sd"],
                                                 [output_dir], num_partitions=num_partitions, dataset_split=split, num_line_skip=1)
    output_config(stats, num_nodes, num_edges, output_dir, "twitter")
    output_config(stats, num_nodes, num_edges, output_dir, "twitter", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "twitter", device = "multi_gpu")


def freebase86m(output_dir, num_partitions=1):
    download_path = download_file("https://data.dgl.ai/dataset/Freebase.zip", output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("Freebase")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("Freebase")).rmdir()

    stats, num_nodes, num_edges = general_parser(
        [str(Path(output_dir) / Path("train.txt")), str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))], ["sdr"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "freebase86m")
    output_config(stats, num_nodes, num_edges, output_dir, "freebase86m", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "freebase86m", device = "multi_gpu")


def wn18(output_dir, num_partitions=1):
    download_path = download_file("https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz",
                                  output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("wordnet-mlj12")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("wordnet-mlj12")).rmdir()

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("wordnet-mlj12-train.txt")),
                                                  str(Path(output_dir) / Path("wordnet-mlj12-valid.txt")),
                                                  str(Path(output_dir) / Path("wordnet-mlj12-test.txt"))], ["srd"],
                                                 [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "wn18")
    output_config(stats, num_nodes, num_edges, output_dir, "wn18", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "wn18", device = "multi_gpu")


def fb15k_237(output_dir, num_partitions=1):
    download_path = download_file("https://data.deepai.org/FB15K-237.2.zip", output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("Release")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("Release")).rmdir()

    stats, num_nodes, num_edges = general_parser(
        [str(Path(output_dir) / Path("train.txt")), str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))], ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k_237")
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k_237", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "fb15k_237", device = "multi_gpu")


def wn18rr(output_dir, num_partitions=1):
    download_path = download_file("https://data.dgl.ai/dataset/wn18rr.zip", output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("wn18rr")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("wn18rr")).rmdir()

    stats, num_nodes, num_edges = general_parser(
        [str(Path(output_dir) / Path("train.txt")), str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))], ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "wn18rr")
    output_config(stats, num_nodes, num_edges, output_dir, "wn18rr", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "wn18rr", device = "multi_gpu")


def codex_s(output_dir, num_partitions=1):
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/train.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/valid.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/test.txt", output_dir)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                  str(Path(output_dir) / Path("valid.txt")),
                                                  str(Path(output_dir) / Path("test.txt"))],
                                                 ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "codex_s")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_s", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_s", device = "multi_gpu")


def codex_m(output_dir, num_partitions=1):
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/train.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/valid.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/test.txt", output_dir)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                  str(Path(output_dir) / Path("valid.txt")),
                                                  str(Path(output_dir) / Path("test.txt"))],
                                                 ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "codex_m")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_m", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_m", device = "multi_gpu")


def codex_l(output_dir, num_partitions=1):
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/train.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/valid.txt", output_dir)
    download_path = download_file(
        "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/test.txt", output_dir)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                  str(Path(output_dir) / Path("valid.txt")),
                                                  str(Path(output_dir) / Path("test.txt"))],
                                                 ["srd"], [output_dir], num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "codex_l")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_l", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "codex_l", device = "multi_gpu")


def drkg(output_dir, num_partitions=1, split=(.05, .05)):
    download_path = download_file("https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz", output_dir)
    extract_file(download_path)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("drkg.tsv"))], ["srd"], [output_dir],
                                                 num_partitions=num_partitions, dataset_split=split)
    output_config(stats, num_nodes, num_edges, output_dir, "drkg")
    output_config(stats, num_nodes, num_edges, output_dir, "drkg", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "drkg", device = "multi_gpu")


def hetionet(output_dir, num_partitions=1, split=(.05, .05)):
    download_path = download_file("https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz",
                                  output_dir)
    extract_file(download_path)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("hetionet-v1.0-edges.sif"))], ["srd"],
                                                 [output_dir], num_partitions=num_partitions, dataset_split=split)
    output_config(stats, num_nodes, num_edges, output_dir, "hetionet")
    output_config(stats, num_nodes, num_edges, output_dir, "hetionet", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "hetionet", device = "multi_gpu")


def kinships(output_dir, num_partitions=1, split=(.05, .05)):
    download_path = download_file("https://archive.ics.uci.edu/ml/machine-learning-databases/kinship/kinship.data",
                                  output_dir)
    edges = []
    pattern = re.compile("^(?P<rel>[a-z]+)\((?P<n1>[A-Za-z]+).{2}(?P<n2>[A-Za-z]+)\)\n$")

    f = open(download_path, "r")
    lines = f.readlines()
    for l in lines:
        if '\n' == l[0]:
            continue
        m = pattern.match(l)
        rel = m.group("rel")
        node_1 = m.group("n1")
        node_2 = m.group("n2")
        edges.append([node_1, rel, node_2])

    if (Path(output_dir) / Path("sample_edges.txt")).exists():
        (Path(output_dir) / Path("sample_edges.txt")).unlink()
    np.random.shuffle(edges)
    np.savetxt((Path(output_dir) / Path("sample_edges.txt")), edges, fmt="%s", delimiter="\t", newline="\n")
    
    stats = stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("sample_edges.txt"))], ["srd"],
                                            [output_dir], dataset_split=split)
    output_config(stats, num_nodes, num_edges, output_dir, "kinships")
    output_config(stats, num_nodes, num_edges, output_dir, "kinships", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "kinships", device = "multi_gpu")


def openbiolink_hq(output_dir, num_partitions=1):
    download_path = download_file("https://zenodo.org/record/3834052/files/HQ_DIR.zip?download=1", output_dir)
    extract_file(download_path)

    stats, num_nodes, num_edges = general_parser(
        [str(Path(output_dir) / Path("HQ_DIR/train_test_data/train_sample.csv")),
         str(Path(output_dir) / Path("HQ_DIR/train_test_data/val_sample.csv")),
         str(Path(output_dir) / Path("HQ_DIR/train_test_data/test_sample.csv"))],
        ["srd"], [output_dir], num_partitions=num_partitions, num_line_skip=0)
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_hq")
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_hq", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_hq", device = "multi_gpu")

def openbiolink_lq(output_dir, num_partitions=1):
    download_path = download_file("https://samwald.info/res/OpenBioLink_2020_final/ALL_DIR.zip", output_dir)
    extract_file(download_path)

    stats, num_nodes, num_edges = general_parser(
        [str(Path(output_dir) / Path("ALL_DIR/train_test_data/train_sample.csv")),
         str(Path(output_dir) / Path("ALL_DIR/train_test_data/val_sample.csv")),
         str(Path(output_dir) / Path("ALL_DIR/train_test_data/test_sample.csv"))],
        ["srd"], [output_dir], num_partitions=num_partitions, num_line_skip=0)
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_lq")
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_lq", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "openbiolink_lq", device = "multi_gpu")


def ogbl_biokg(output_dir, num_partitions=1):
    download_path = download_file("https://snap.stanford.edu/ogb/data/linkproppred/biokg.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("biokg/split/random/train.pt")),
             str(Path(output_dir) / Path("biokg/split/random/valid.pt")),
             str(Path(output_dir) / Path("biokg/split/random/test.pt"))]

    stats, num_nodes, num_edges = parse_ogbl(files, True, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_biokg")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_biokg", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_biokg", device = "multi_gpu")


def ogbl_ppa(output_dir, num_partitions=1):
    download_path = download_file("https://snap.stanford.edu/ogb/data/linkproppred/ppassoc.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("ppassoc/split/throughput/train.pt")),
             str(Path(output_dir) / Path("ppassoc/split/throughput/valid.pt")),
             str(Path(output_dir) / Path("ppassoc/split/throughput/test.pt"))]

    stats, num_nodes, num_edges = parse_ogbl(files, False, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ppa")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ppa", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ppa", device = "multi_gpu")


def ogbl_ddi(output_dir, num_partitions=1):
    download_path = download_file("https://snap.stanford.edu/ogb/data/linkproppred/ddi.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("ddi/split/target/train.pt")),
             str(Path(output_dir) / Path("ddi/split/target/valid.pt")),
             str(Path(output_dir) / Path("ddi/split/target/test.pt"))]

    stats, num_nodes, num_edges = parse_ogbl(files, False, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ddi")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ddi", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_ddi", device = "multi_gpu")


def ogbl_collab(output_dir, num_partitions=1):
    download_path = download_file("https://snap.stanford.edu/ogb/data/linkproppred/collab.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("collab/split/time/train.pt")),
             str(Path(output_dir) / Path("collab/split/time/valid.pt")),
             str(Path(output_dir) / Path("collab/split/time/test.pt"))]

    stats, num_nodes, num_edges = parse_ogbl(files, False, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_collab")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_collab", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbl_collab", device = "multi_gpu")


def ogbn_arxiv(output_dir, num_partitions=1):
    download_path = download_file("http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip	", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("arxiv/split/time/train.csv.gz")),
             str(Path(output_dir) / Path("arxiv/split/time/valid.csv.gz")),
             str(Path(output_dir) / Path("arxiv/split/time/test.csv.gz")),
             str(Path(output_dir) / Path("arxiv/raw/edge.csv.gz"))]

    stats, num_nodes, num_edges = parse_ogbn(files, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_arxiv")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_arxiv", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_arxiv", device = "multi_gpu")


def ogbn_proteins(output_dir, num_partitions=1):
    download_path = download_file("http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("proteins/split/species/train.csv.gz")),
             str(Path(output_dir) / Path("proteins/split/species/valid.csv.gz")),
             str(Path(output_dir) / Path("proteins/split/species/test.csv.gz")),
             str(Path(output_dir) / Path("proteins/raw/edge.csv.gz"))]

    stats, num_nodes, num_edges = parse_ogbn(files, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_proteins")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_proteins", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_proteins", device = "multi_gpu")

def ogbn_products(output_dir, num_partitions=1):
    download_path = download_file("http://snap.stanford.edu/ogb/data/nodeproppred/products.zip", output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("products/split/sales_ranking/train.csv.gz")),
             str(Path(output_dir) / Path("products/split/sales_ranking/valid.csv.gz")),
             str(Path(output_dir) / Path("products/split/sales_ranking/test.csv.gz")),
             str(Path(output_dir) / Path("products/raw/edge.csv.gz"))]

    stats, num_nodes, num_edges = parse_ogbn(files, output_dir, num_partitions=num_partitions)
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_products")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_products", device = "gpu")
    output_config(stats, num_nodes, num_edges, output_dir, "ogbn_products", device = "multi_gpu")

def parse_ogbn(files, output_dir, num_partitions=1):
    splits = []
    for file in files[0:-1]:
        nodes = pd.read_csv(file, compression='gzip', header=None)
        splits.append(nodes)

    edges = pd.read_csv(files[-1], compression='gzip', header=None)

    train_edges = edges.loc[np.in1d(edges[0], splits[0])]
    valid_edges = edges.loc[np.in1d(edges[0], splits[1])]
    test_edges = edges.loc[np.in1d(edges[0], splits[2])]

    train_edges.to_csv(str(Path(output_dir) / Path("train.txt")), sep="\t", header=False, index=False)
    valid_edges.to_csv(str(Path(output_dir) / Path("valid.txt")), sep="\t", header=False, index=False)
    test_edges.to_csv(str(Path(output_dir) / Path("test.txt")), sep="\t", header=False, index=False)

    stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                  str(Path(output_dir) / Path("valid.txt")),
                                                  str(Path(output_dir) / Path("test.txt"))], ["sd"], [output_dir],
                                                 num_partitions=num_partitions)
    return stats, num_nodes, num_edges

def parse_ogbl(files, has_rel, output_dir, num_partitions=1):
    if has_rel == True:
        train_idx = torch.load(str(files[0]))
        valid_idx = torch.load(str(files[1]))
        test_idx = torch.load(str(files[2]))
        train_list = np.array([train_idx.get("head"), train_idx.get("relation"), train_idx.get("tail")]).T
        valid_list = np.array([valid_idx.get("head"), valid_idx.get("relation"), valid_idx.get("tail")]).T
        test_list = np.array([test_idx.get("head"), test_idx.get("relation"), test_idx.get("tail")]).T
    else:
        train_list = torch.load(files[0]).get("edge")
        valid_list = torch.load(files[1]).get("edge")
        test_list = torch.load(files[2]).get("edge")

    np.savetxt(str(Path(output_dir) / Path("train.txt")), train_list, fmt="%s", delimiter="\t", newline="\n")
    np.savetxt(str(Path(output_dir) / Path("valid.txt")), valid_list, fmt="%s", delimiter="\t", newline="\n")
    np.savetxt(str(Path(output_dir) / Path("test.txt")), test_list, fmt="%s", delimiter="\t", newline="\n")
    print("Conversion completed.")

    if has_rel == True:
        stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                      str(Path(output_dir) / Path("valid.txt")),
                                                      str(Path(output_dir) / Path("test.txt"))], ["srd"],
                                                     [output_dir], num_partitions=num_partitions)
    else:
        stats, num_nodes, num_edges = general_parser([str(Path(output_dir) / Path("train.txt")),
                                                      str(Path(output_dir) / Path("valid.txt")),
                                                      str(Path(output_dir) / Path("test.txt"))], ["sd"],
                                                     [output_dir], num_partitions=num_partitions)
    return stats, num_nodes, num_edges


def download_file(url, output_dir):
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    url_components = urlparse(url)
    filename = Path(url_components.path + url_components.query).name
    filepath = output_dir / filename

    if filepath.is_file():
        print(f"File already exists: {filepath}")
    else:
        try:
            print(f"Downloading {filename} to {filepath}")
            urlretrieve(url, str(filepath))
        except OSError:
            raise RuntimeError(f"Failed to download {filename}")

    return filepath


def extract_file(filepath):
    print("Extracting")
    if tarfile.is_tarfile(str(filepath)):
        if str(filepath).endswith(".gzip") or str(filepath).endswith(".gz"):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=filepath.parent)
        elif str(filepath).endswith(".tar.gz") or str(filepath).endswith(".tgz"):
            with tarfile.open(filepath, "r:gz") as tar:
                tar.extractall(path=filepath.parent)
        elif str(filepath).endswith(".tar"):
            with tarfile.open(filepath, "r:") as tar:
                tar.extractall(path=filepath.parent)
        elif str(filepath).endswith(".bz2"):
            with tarfile.open(filepath, "r:bz2") as tar:
                tar.extractall(path=filepath.parent)
        else:
            try:
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            except tarfile.TarError:
                raise RuntimeError("Unrecognized file format, need to extract and call general converter manually.")
    elif zipfile.is_zipfile(str(filepath)):
        with ZipFile(filepath, "r") as zip:
            zip.extractall(filepath.parent)
    else:
        try:
            with filepath.with_suffix("").open("wb") as output_f, gzip.GzipFile(filepath) as gzip_f:
                shutil.copyfileobj(gzip_f, output_f)
        except gzip.BadGzipFile:
            raise RuntimeError("Undefined file format.")
        except:
            raise RuntimeError("Undefined exception.")

    if filepath.exists():
        filepath.unlink()

    print("Extraction completed")
    return filepath.parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess Datasets')
    parser.add_argument('dataset', metavar='dataset', type=str, help='Dataset to preprocess')
    parser.add_argument('output_directory', metavar='output_directory', type=str, help='Directory to put graph data')
    parser.add_argument('--num_partitions', metavar='num_partitions', required=False, type=int, default=1,
                        help='Number of partitions to split the edges into')

    args = parser.parse_args()

    try:
        if not Path(args.output_directory).exists():
            Path(args.output_directory).mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Directory already exists.")

    print(args.dataset)

    dataset_dict = {
        "twitter": twitter,
        "fb15k": fb15k,
        "live_journal": live_journal,
        "freebase86m": freebase86m,
        "wn18": wn18,
        "fb15k_237": fb15k_237,
        "wn18rr": wn18rr,
        "codex_s": codex_s,
        "codex_m": codex_m,
        "codex_l": codex_l,
        "drkg": drkg,
        "hetionet": hetionet,
        "kinships": kinships,
        "openbiolink_hq": openbiolink_hq,
        "openbiolink_lq": openbiolink_lq,
        "ogbl_biokg": ogbl_biokg,
        "ogbl_ppa": ogbl_ppa,
        "ogbl_ddi": ogbl_ddi,
        "ogbl_collab": ogbl_collab,
        "ogbn_arxiv": ogbn_arxiv,
        "ogbn_proteins": ogbn_proteins,
        "ogbn_products": ogbn_products,
    }

    if dataset_dict.get(args.dataset) != None:
        dataset_dict.get(args.dataset)(args.output_directory, args.num_partitions)
        output_bash_cmds(args.output_directory, args.dataset)
    else:
        print("Unrecognized dataset!")


