"""Preprocess module of Marius.

This module contains the functions for preprocessing both custom datasets and
supported datasets.
"""

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

from marius.tools.config_generator import output_config
from marius.tools.config_generator import read_template
from marius.tools.config_generator import set_up_files
from marius.tools.config_generator import update_stats
from marius.tools.config_generator import update_data_path
from marius.tools.config_generator import DEFAULT_CONFIG_FILE
from marius.tools.csv_converter import general_parser


def live_journal(output_dir, num_partitions=1, split=(.05, .05)):
    """Preprocesses the dataset live_journal.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.
        split: The proportion of the validation and test sections.

    Returns:
        The statistics of current dataset.
    """
    LIVE_JOURNAL_URL = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
    download_path = download_file(LIVE_JOURNAL_URL, output_dir)
    extract_file(download_path)
    return general_parser([str(Path(output_dir) /
                          Path("soc-LiveJournal1.txt"))], ["sd"],
                          [output_dir], num_partitions=num_partitions,
                          dataset_split=split)


def fb15k(output_dir, num_partitions=1):
    """Preprocesses the dataset fb15k.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    FB15K_URL = "https://dl.fbaipublicfiles.com/starspace/fb15k.tgz"
    download_path = download_file(FB15K_URL, output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("FB15k")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("FB15k")).rmdir()

    return general_parser(
            [str(Path(output_dir) /
             Path("freebase_mtr100_mte100-train.txt")),
             str(Path(output_dir) / Path("freebase_mtr100_mte100-valid.txt")),
             str(Path(output_dir) / Path("freebase_mtr100_mte100-test.txt"))],
            ["srd"], [output_dir], num_partitions=num_partitions)


def twitter(output_dir, num_partitions=1, split=(.05, .05)):
    """Preprocesses the dataset twitter.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.
        split: The proportion of the validation and test sections.

    Returns:
        The statistics of current dataset.
    """
    TWITTER_URL = "https://snap.stanford.edu/data/twitter-2010.txt.gz"
    download_path = download_file(TWITTER_URL, output_dir)
    extract_file(download_path)

    return general_parser([str(Path(output_dir) / Path("twitter-2010.txt"))],
                          ["srd"],
                          [output_dir], num_partitions=num_partitions,
                          dataset_split=split, num_line_skip=1)


def freebase86m(output_dir, num_partitions=1):
    """Preprocesses the dataset freebase86m.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    FREEBASE86M_URL = "https://data.dgl.ai/dataset/Freebase.zip"
    download_path = download_file(FREEBASE86M_URL, output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("Freebase")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("Freebase")).rmdir()

    return general_parser(
        [str(Path(output_dir) / Path("train.txt")),
         str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))],
        ["sdr"],
        [output_dir], num_partitions=num_partitions)


def wn18(output_dir, num_partitions=1):
    """Preprocesses the dataset wn18.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    WN18_URL = "https://everest.hds.utc.fr/lib/exe/fetch.php?media=en:wordnet-mlj12.tar.gz"
    download_path = download_file(WN18_URL, output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("wordnet-mlj12")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("wordnet-mlj12")).rmdir()

    return general_parser(
            [str(Path(output_dir) / Path("wordnet-mlj12-train.txt")),
             str(Path(output_dir) / Path("wordnet-mlj12-valid.txt")),
             str(Path(output_dir) / Path("wordnet-mlj12-test.txt"))], ["srd"],
            [output_dir], num_partitions=num_partitions)


def fb15k_237(output_dir, num_partitions=1):
    """Preprocesses the dataset fb15k_237.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    FB15K_237 = "https://data.deepai.org/FB15K-237.2.zip"
    download_path = download_file(FB15K_237, output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("Release")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("Release")).rmdir()

    return general_parser(
        [str(Path(output_dir) / Path("train.txt")),
         str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))],
        ["srd"], [output_dir], num_partitions=num_partitions)


def wn18rr(output_dir, num_partitions=1):
    """Preprocesses the dataset wn18rr.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    WN18RR_URL = "https://data.dgl.ai/dataset/wn18rr.zip"
    download_path = download_file(WN18RR_URL, output_dir)
    extract_file(download_path)
    for file in (output_dir / Path("wn18rr")).iterdir():
        file.rename(output_dir / Path(file.name))
    (output_dir / Path("wn18rr")).rmdir()

    return general_parser(
        [str(Path(output_dir) / Path("train.txt")),
         str(Path(output_dir) / Path("valid.txt")),
         str(Path(output_dir) / Path("test.txt"))],
        ["srd"], [output_dir], num_partitions=num_partitions)


def codex_s(output_dir, num_partitions=1):
    """Preprocesses the dataset codex_s.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    CODEX_S_TRAIN_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/train.txt"
    CODEX_S_VALID_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/valid.txt"
    CODEX_S_TEST_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-s/test.txt"

    download_path = download_file(CODEX_S_TRAIN_URL, output_dir)
    download_path = download_file(CODEX_S_VALID_URL, output_dir)
    download_path = download_file(CODEX_S_TEST_URL, output_dir)

    return general_parser([str(Path(output_dir) / Path("train.txt")),
                           str(Path(output_dir) / Path("valid.txt")),
                           str(Path(output_dir) / Path("test.txt"))],
                          ["srd"], [output_dir],
                          num_partitions=num_partitions)


def codex_m(output_dir, num_partitions=1):
    """Preprocesses the dataset codex_m.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    CODEX_M_TRAIN_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/train.txt"
    CODEX_M_VALID_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/valid.txt"
    CODEX_M_TEST_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-m/test.txt"
    download_path = download_file(CODEX_M_TRAIN_URL, output_dir)
    download_path = download_file(CODEX_M_VALID_URL, output_dir)
    download_path = download_file(CODEX_M_TEST_URL, output_dir)

    return general_parser([str(Path(output_dir) / Path("train.txt")),
                           str(Path(output_dir) / Path("valid.txt")),
                           str(Path(output_dir) / Path("test.txt"))],
                          ["srd"], [output_dir], num_partitions=num_partitions)


def codex_l(output_dir, num_partitions=1):
    """Preprocesses the dataset codex_l.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    CODEX_L_TRAIN_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/train.txt"
    CODEX_L_VALID_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/valid.txt"
    CODEX_L_TEST_URL = "https://raw.githubusercontent.com/tsafavi/codex/master/data/triples/codex-l/test.txt"
    download_path = download_file(CODEX_L_TRAIN_URL, output_dir)
    download_path = download_file(CODEX_L_VALID_URL, output_dir)
    download_path = download_file(CODEX_L_TEST_URL, output_dir)

    return general_parser([str(Path(output_dir) / Path("train.txt")),
                           str(Path(output_dir) / Path("valid.txt")),
                           str(Path(output_dir) / Path("test.txt"))],
                          ["srd"], [output_dir], num_partitions=num_partitions)


def drkg(output_dir, num_partitions=1, split=(.05, .05)):
    """Preprocesses the dataset drkg.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.
        split: The proportion of the validation and test sections.

    Returns:
        The statistics of current dataset.
    """
    DRKG_URL = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
    download_path = download_file(DRKG_URL, output_dir)
    extract_file(download_path)

    return general_parser([str(Path(output_dir) /
                          Path("drkg.tsv"))], ["srd"], [output_dir],
                          num_partitions=num_partitions, dataset_split=split)


def hetionet(output_dir, num_partitions=1, split=(.05, .05)):
    """Preprocesses the dataset hetionet.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.
        split: The proportion of the validation and test sections.

    Returns:
        The statistics of current dataset.
    """
    HETIONET_URL = "https://github.com/hetio/hetionet/raw/master/hetnet/tsv/hetionet-v1.0-edges.sif.gz"
    download_path = download_file(HETIONET_URL, output_dir)
    extract_file(download_path)

    return general_parser([str(Path(output_dir) /
                           Path("hetionet-v1.0-edges.sif"))], ["srd"],
                          [output_dir], num_partitions=num_partitions,
                          dataset_split=split)


def kinships(output_dir, num_partitions=1, split=(.05, .05)):
    """Preprocesses the dataset kinships.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.
        split: The proportion of the validation and test sections.

    Returns:
        The statistics of current dataset.
    """
    KINSHIPS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/kinship/kinship.data"
    download_path = download_file(KINSHIPS_URL, output_dir)
    edges = []
    pattern = re.compile("^(?P<rel>[a-z]+)" +
                         r"\((?P<n1>[A-Za-z]+).{2}(?P<n2>[A-Za-z]+)\)\n$")

    f = open(download_path, "r")
    lines = f.readlines()
    for line in lines:
        if '\n' == line[0]:
            continue
        m = pattern.match(line)
        rel = m.group("rel")
        node_1 = m.group("n1")
        node_2 = m.group("n2")
        edges.append([node_1, rel, node_2])

    if (Path(output_dir) / Path("sample_edges.txt")).exists():
        (Path(output_dir) / Path("sample_edges.txt")).unlink()
    np.random.shuffle(edges)
    np.savetxt((Path(output_dir) / Path("sample_edges.txt")), edges, fmt="%s",
               delimiter="\t", newline="\n")

    return general_parser([str(Path(output_dir) / Path("sample_edges.txt"))],
                          ["srd"], [output_dir], dataset_split=split)


def openbiolink_hq(output_dir, num_partitions=1):
    """Preprocesses the dataset openbiolink_hq.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OPENBIOLINK_HQ_URL = "https://zenodo.org/record/3834052/files/HQ_DIR.zip?download=1"
    download_path = download_file(OPENBIOLINK_HQ_URL, output_dir)
    extract_file(download_path)

    return general_parser(
        [str(Path(output_dir) /
         Path("HQ_DIR/train_test_data/train_sample.csv")),
         str(Path(output_dir) /
         Path("HQ_DIR/train_test_data/val_sample.csv")),
         str(Path(output_dir) /
         Path("HQ_DIR/train_test_data/test_sample.csv"))],
        ["srd"], [output_dir], num_partitions=num_partitions, num_line_skip=0)


def openbiolink_lq(output_dir, num_partitions=1):
    """Preprocesses the dataset openbiolink_lq.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OPENBIOLINK_LQ_URL = "https://samwald.info/res/OpenBioLink_2020_final/ALL_DIR.zip"
    download_path = download_file(OPENBIOLINK_LQ_URL, output_dir)
    extract_file(download_path)

    return general_parser(
        [str(Path(output_dir) /
         Path("ALL_DIR/train_test_data/train_sample.csv")),
         str(Path(output_dir) /
         Path("ALL_DIR/train_test_data/val_sample.csv")),
         str(Path(output_dir) /
         Path("ALL_DIR/train_test_data/test_sample.csv"))],
        ["srd"], [output_dir], num_partitions=num_partitions, num_line_skip=0)


def ogbl_biokg(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbl_biokg.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBL_BIOKG_URL = "https://snap.stanford.edu/ogb/data/linkproppred/biokg.zip"
    download_path = download_file(OGBL_BIOKG_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("biokg/split/random/train.pt")),
             str(Path(output_dir) / Path("biokg/split/random/valid.pt")),
             str(Path(output_dir) / Path("biokg/split/random/test.pt"))]

    return parse_ogbl(files, True, output_dir, num_partitions=num_partitions)


def ogbl_ppa(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbl_ppa.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBL_PPA_URL = "https://snap.stanford.edu/ogb/data/linkproppred/ppassoc.zip"
    download_path = download_file(OGBL_PPA_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("ppassoc/split/throughput/train.pt")),
             str(Path(output_dir) / Path("ppassoc/split/throughput/valid.pt")),
             str(Path(output_dir) / Path("ppassoc/split/throughput/test.pt"))]

    return parse_ogbl(files, False, output_dir, num_partitions=num_partitions)


def ogbl_ddi(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbl_ddi.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBL_DDI_URL = "https://snap.stanford.edu/ogb/data/linkproppred/ddi.zip"
    download_path = download_file(OGBL_DDI_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("ddi/split/target/train.pt")),
             str(Path(output_dir) / Path("ddi/split/target/valid.pt")),
             str(Path(output_dir) / Path("ddi/split/target/test.pt"))]

    return parse_ogbl(files, False, output_dir, num_partitions=num_partitions)


def ogbl_collab(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbl_collab.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBL_COLLAB_URL = "https://snap.stanford.edu/ogb/data/linkproppred/collab.zip"
    download_path = download_file(OGBL_COLLAB_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("collab/split/time/train.pt")),
             str(Path(output_dir) / Path("collab/split/time/valid.pt")),
             str(Path(output_dir) / Path("collab/split/time/test.pt"))]

    return parse_ogbl(files, False, output_dir, num_partitions=num_partitions)


def ogbn_arxiv(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbn_arxiv.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBN_ARXIV_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"
    download_path = download_file(OGBN_ARXIV_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) / Path("arxiv/split/time/train.csv.gz")),
             str(Path(output_dir) / Path("arxiv/split/time/valid.csv.gz")),
             str(Path(output_dir) / Path("arxiv/split/time/test.csv.gz")),
             str(Path(output_dir) / Path("arxiv/raw/edge.csv.gz"))]

    return parse_ogbn(files, output_dir, num_partitions=num_partitions)


def ogbn_proteins(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbn_proteins.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBN_PROTEINS_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/proteins.zip"
    download_path = download_file(OGBN_PROTEINS_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) /
             Path("proteins/split/species/train.csv.gz")),
             str(Path(output_dir) /
             Path("proteins/split/species/valid.csv.gz")),
             str(Path(output_dir) /
             Path("proteins/split/species/test.csv.gz")),
             str(Path(output_dir) / Path("proteins/raw/edge.csv.gz"))]

    return parse_ogbn(files, output_dir, num_partitions=num_partitions)


def ogbn_products(output_dir, num_partitions=1):
    """Preprocesses the dataset ogbn_products.

    Args:
        output_dir: The directory where the original dataset files and
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    OGBN_PRODUCTS_URL = "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip"
    download_path = download_file(OGBN_PRODUCTS_URL, output_dir)
    extract_file(download_path)
    files = [str(Path(output_dir) /
             Path("products/split/sales_ranking/train.csv.gz")),
             str(Path(output_dir) /
             Path("products/split/sales_ranking/valid.csv.gz")),
             str(Path(output_dir) /
             Path("products/split/sales_ranking/test.csv.gz")),
             str(Path(output_dir) / Path("products/raw/edge.csv.gz"))]

    return parse_ogbn(files, output_dir, num_partitions=num_partitions)


def parse_ogbn(files, output_dir, num_partitions=1):
    """Parse ogbn datasets.

    Retrieves the graph data from downloaded ogbn dataset files.

    Args:
        files: The original ogbn dataset files.
        output_dir: The directory where the original dataset files and 
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    splits = []
    for file in files[0:-1]:
        nodes = pd.read_csv(file, compression='gzip', header=None)
        splits.append(nodes)

    edges = pd.read_csv(files[-1], compression='gzip', header=None)

    train_edges = edges.loc[np.in1d(edges[0], splits[0])]
    valid_edges = edges.loc[np.in1d(edges[0], splits[1])]
    test_edges = edges.loc[np.in1d(edges[0], splits[2])]

    train_edges.to_csv(str(Path(output_dir) /
                       Path("train.txt")), sep="\t", header=False, index=False)
    valid_edges.to_csv(str(Path(output_dir) /
                       Path("valid.txt")), sep="\t", header=False, index=False)
    test_edges.to_csv(str(Path(output_dir) /
                      Path("test.txt")), sep="\t", header=False, index=False)

    stats = general_parser(
                    [str(Path(output_dir) / Path("train.txt")),
                     str(Path(output_dir) / Path("valid.txt")),
                     str(Path(output_dir) / Path("test.txt"))],
                    ["sd"], [output_dir],
                    num_partitions=num_partitions)
    return stats


def parse_ogbl(files, has_rel, output_dir, num_partitions=1):
    """Parse ogbl datasets.

    Retrieves the graph from downloaded ogbl dataset files.

    Args:
        files: The original obgl dataset files.
        has_rel: Indicates whether the current dataset has relation edges.
        output_dir: The directory where the original dataset files and 
            preprocessed files will be stored.
        num_partitions: The number of data partitions.

    Returns:
        The statistics of current dataset.
    """
    if has_rel is True:
        train_idx = torch.load(str(files[0]))
        valid_idx = torch.load(str(files[1]))
        test_idx = torch.load(str(files[2]))
        train_list = np.array([train_idx.get("head"),
                               train_idx.get("relation"),
                               train_idx.get("tail")]).T
        valid_list = np.array([valid_idx.get("head"),
                               valid_idx.get("relation"),
                               valid_idx.get("tail")]).T
        test_list = np.array([test_idx.get("head"),
                              test_idx.get("relation"),
                              test_idx.get("tail")]).T
    else:
        train_list = torch.load(files[0]).get("edge")
        valid_list = torch.load(files[1]).get("edge")
        test_list = torch.load(files[2]).get("edge")

    np.savetxt(str(Path(output_dir) / Path("train.txt")),
               train_list, fmt="%s", delimiter="\t", newline="\n")
    np.savetxt(str(Path(output_dir) / Path("valid.txt")),
               valid_list, fmt="%s", delimiter="\t", newline="\n")
    np.savetxt(str(Path(output_dir) / Path("test.txt")),
               test_list, fmt="%s", delimiter="\t", newline="\n")
    print("Conversion completed.")

    if has_rel is True:
        stats = general_parser(
            [str(Path(output_dir) / Path("train.txt")),
             str(Path(output_dir) / Path("valid.txt")),
             str(Path(output_dir) / Path("test.txt"))], ["srd"],
            [output_dir], num_partitions=num_partitions)
    else:
        stats = general_parser(
            [str(Path(output_dir) / Path("train.txt")),
             str(Path(output_dir) / Path("valid.txt")),
             str(Path(output_dir) / Path("test.txt"))], ["sd"],
            [output_dir], num_partitions=num_partitions)
    return stats


def download_file(url, output_dir):
    """Downloads files.

    Downloads the files to designated directory with given url.

    Args:
        url: The url to the files to be downloaded.
        output_dir: The location to save all downloaded files.

    Returns:
        The path to the downloaded files.

    Raises:
        RuntimeError: An error occurred when downloading is failed.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir()

    url_components = urlparse(url)
    filename = Path(url_components.path + url_components.query).name
    filepath = output_dir / filename

    if filepath.is_file():
        print(f"File already exists: {filepath} May be outdated!")

    else:
        try:
            print(f"Downloading {filename} to {filepath}")
            urlretrieve(url, str(filepath))
        except OSError:
            raise RuntimeError(f"Failed to download {filename}")

    return filepath


def extract_file(filepath):
    """Extracts files.

    Extracts the files pointed by filepath.

    Args:
        filepath: The path to the files needed to be extracted.

    Returns:
        The directory contains all extracted files.

    Raises:
        RuntimeError: An error occurred when the file format cannot be 
            recognized or the file to be extracted is not 
            complete. Detailed information is given if the exception is raised.
        
    """
    print("Extracting")
    try:
        if tarfile.is_tarfile(str(filepath)):
            if (str(filepath).endswith(".gzip") or
                    str(filepath).endswith(".gz")):
                with tarfile.open(filepath, "r:gz") as tar:
                    tar.extractall(path=filepath.parent)
            elif (str(filepath).endswith(".tar.gz") or
                    str(filepath).endswith(".tgz")):
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
                    raise RuntimeError(
                        "Unrecognized file format, need to " +
                        "extract and call general converter manually.")
        elif zipfile.is_zipfile(str(filepath)):
            with ZipFile(filepath, "r") as zip:
                zip.extractall(filepath.parent)
        else:
            try:
                with filepath.with_suffix("").open("wb") as output_f, \
                     gzip.GzipFile(filepath) as gzip_f:
                    shutil.copyfileobj(gzip_f, output_f)
            except gzip.BadGzipFile:
                raise RuntimeError("Undefined file format.")
            except:
                raise RuntimeError("Undefined exception.")
    except EOFError:
        raise RuntimeError("Dataset file isn't complete. Try download again.")

    if filepath.exists():
        filepath.unlink()

    print("Extraction completed")
    return filepath.parent


def update_param(config_dict, arg_dict):
    """Updates parametars.

    Updates parameters for the configuration files to be generated according to
        command line arguments.

    Args:
        config_dict: The dict containing all configuration parameters and their
            default values.
        arg_dict: The dict containing all command line arguments.

    Returns:
        The updated configuration dict.
    """
    if arg_dict.get("generate_config") is None:
        for key in config_dict:
            if arg_dict.get(key) is not None:
                raise RuntimeError(
                    "Please specify --generate_config when " +
                    "specifying generating options"
                )
    else:
        if arg_dict.get("generate_config") is None:
            config_dict.update({"device": "GPU"})
            config_dict.update({"general.device": "GPU"})
        elif arg_dict.get("generate_config") == "multi-GPU":
            config_dict.update({"device": "multi_GPU"})
            config_dict.update({"general.device": "multi-GPU"})
        else:
            config_dict.update({"general.device":
                                arg_dict.get("generate_config")})
            config_dict.update({"device":
                                arg_dict.get("generate_config")})

        for key in config_dict.keys():
            if arg_dict.get(key) is not None:
                config_dict.update({key: arg_dict.get(key)})

    if config_dict.get("general.random_seed") == "#":
        config_dict.pop("general.random_seed")

    return config_dict


def set_args():
    """Sets command line arguments for this preprocess module.

    Returns:
        The parser containing all command line arguments and the configuration
            dict containing all parameters and their default values.
    """
    parser = argparse.ArgumentParser(
                description='Preprocess Datasets', prog='preprocess',
                formatter_class=argparse.RawTextHelpFormatter,
                epilog=(('Specify certain config (optional): ' +
                        '[--<section>.<key>=<value>]')))
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('output_directory', metavar='output_directory',
                        type=str, help='Directory to put graph data')
    mode.add_argument('--files', metavar='files', nargs='+', type=str,
                        help='Files containing custom dataset')
    mode.add_argument('--dataset', metavar='dataset',
                        type=str, help='Supported dataset to preprocess')
    parser.add_argument('--num_partitions', metavar='num_partitions',
                        required=False, type=int, default=1,
                        help='Number of partitions to split the edges into')
    parser.add_argument('--overwrite', action='store_true',
                        required=False,
                        help=('Overwrites the output_directory if this is ' +
                              'set. '
                              'Otherwise, files with same the names will be ' +
                              'treated as the data for current dataset.'))
    parser.add_argument('--generate_config', '-gc', metavar='generate_config',
                        choices=["GPU", "CPU", "multi-GPU"],
                        nargs='?', const="GPU",
                        help=('Generates a single-GPU ' +
                              'training configuration file by default. ' +
                              '\nValid options (default to GPU): ' +
                              '[GPU, CPU, multi-GPU]'))
    parser.add_argument('--format', metavar='format', nargs=1, type=str,
                        default=['srd'],
                        help='Format of data, eg. srd')
    parser.add_argument('--delim', '-d', metavar='delim', type=str,
                        default="",
                        help='Specifies the delimiter')
    parser.add_argument('--dtype', metavar='dtype', type=np.dtype,
                        default=np.int32,
                        help='Indicates the numpy.dtype')
    parser.add_argument('--not_remap_ids', action='store_false',
                        help='If set, will not remap ids')
    parser.add_argument('--dataset_split', '-ds', metavar='dataset_split',
                        nargs=2, type=float, default=(-1, -1),
                        help='Split dataset into specified fractions')
    parser.add_argument('--start_col', '-sc', metavar='start_col', type=int,
                        default=0,
                        help='Indicates the column index to start from')
    parser.add_argument('--num_line_skip', '-nls', metavar='num_line_skip',
                        type=int, default=None,
                        help='Indicates number of lines to ' +
                             'skip from the beginning')

    config_dict, valid_dict = read_template(DEFAULT_CONFIG_FILE)

    for key in list(config_dict.keys())[1:]:
        if valid_dict.get(key) is not None:
            parser.add_argument(str("--" + key), metavar=key, type=str,
                                choices=valid_dict.get(key),
                                help=argparse.SUPPRESS)
        else:
            parser.add_argument(str("--" + key), metavar=key, type=str,
                                help=argparse.SUPPRESS)

    return parser, config_dict


def parse_args(config_dict, args):
    """Parse command line arguments.

    Identifies the dataset to be preprocess and update configuration parameters
        if they are set by command line arguments.

    Args:
        config_dict: The dict containing all configuration parameters and their
            default values.
        args: All command line arguments.

    Returns:
        The dict containing updated configuration parameters and the dict
            containing parsed command line arguments. 
    """
    arg_dict = vars(args)
    config_dict = update_param(config_dict, arg_dict)
    set_up_files(args.output_directory)
    
    if arg_dict.get("dataset") is None:
        config_dict.update({"dataset": "custom"})
    else:
        config_dict.update({"dataset": arg_dict.get("dataset")})
    
    return config_dict, arg_dict


def main():
    parser, config_dict = set_args()
    args = parser.parse_args()
    config_dict, arg_dict = parse_args(config_dict, args)

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

    if args.overwrite and Path(args.output_directory).exists():
        shutil.rmtree(args.output_directory)

    if dataset_dict.get(args.dataset) is not None:
        print(args.dataset)
        stats = dataset_dict.get(args.dataset)(
                                    args.output_directory, args.num_partitions)
    else:
        print("Preprocess custom dataset")
        stats = general_parser(args.files, args.format,
                               args.output_directory, args.delim,
                               args.num_partitions,
                               args.dtype, args.not_remap_ids,
                               args.dataset_split,
                               args.start_col,
                               args.num_line_skip)


    if args.generate_config is not None:
        dir = args.output_directory
        config_dict = update_stats(stats, config_dict)
        config_dict = update_data_path(dir, config_dict)
        output_config(config_dict, dir)

if __name__ == "__main__":
    main()
