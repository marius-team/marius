from marius.tools.preprocess import download_file, extract_file
from marius.tools.csv_converter import split_dataset
from pathlib import Path
import pandas as pd
import numpy as np

def preproccess_live_journal():

    output_dir = "live_journal_dglke"

    LIVE_JOURNAL_URL = "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
    download_path = download_file(LIVE_JOURNAL_URL, output_dir)
    extract_file(download_path)

    filename = str(Path(output_dir) / Path("soc-LiveJournal1.txt"))

    validation_fraction = .05
    test_fraction = .05
    train_fraction = 1 - validation_fraction - test_fraction

    assert(train_fraction > 0)
    assert (validation_fraction + test_fraction > 0)
    train_file = output_dir + "/train_edges.txt"
    valid_file = output_dir + "/valid_edges.txt"
    test_file = output_dir + "/test_edges.txt"

    chunk_size = 10**7
    with open(train_file, "a") as f, open(valid_file, "a") as g, open(test_file, "a") as h:
        for chunk in pd.read_csv(filename, sep="\t", header=None,
                                 chunksize=chunk_size,
                                 skiprows=4,
                                 dtype=str):

            src_nodes = np.asarray(chunk[chunk.columns.values[0]])
            dst_nodes = np.asarray(chunk[chunk.columns.values[1]])
            edges = np.stack([src_nodes, np.zeros_like(src_nodes), dst_nodes]).T

            train, valid, test = np.split(
                edges,
                [int(train_fraction*len(chunk)),
                 int((train_fraction + validation_fraction)*len(chunk))])

            edges = np.stack([src_nodes, np.zeros_like(src_nodes),
                              dst_nodes]).T

            train = np.asarray(train, dtype=np.str_)
            valid = np.asarray(valid, dtype=np.str_)
            test = np.asarray(test, dtype=np.str_)

            np.savetxt(f, train, fmt='%s', delimiter="\t")
            np.savetxt(g, valid, fmt='%s', delimiter="\t")
            np.savetxt(h, test, fmt='%s', delimiter="\t")


