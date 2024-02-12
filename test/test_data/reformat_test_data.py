import argparse
import os

import numpy as np
import pandas as pd


def load_txt_as_df(file_path):
    return pd.read_csv(file_path, sep=" ", header=None, names=["src", "edge_type", "dst"])


def main():
    train_df = load_txt_as_df("train_edges.txt")
    test_df = load_txt_as_df("test_edges.txt")
    valid_df = load_txt_as_df("valid_edges.txt")

    combined_df = pd.concat([train_df, test_df, valid_df])
    combined_df["edge_weights"] = np.random.uniform(-1.0, 1.0, len(combined_df.index))
    combined_df.to_csv("test_edges_with_weights.csv", index=False, header=False)


if __name__ == "__main__":
    main()
