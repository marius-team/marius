import argparse
import re
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import itertools


def split_dataset(input_dataset, validation_fraction, test_fraction,
                  entry_regex, num_line_skip, data_cols,
                  delim, dtype=np.int32):
    train_fraction = 1 - validation_fraction - test_fraction

    assert(train_fraction > 0)
    assert (validation_fraction + test_fraction > 0)
    base_path = "/".join(input_dataset.split("/")[:-1])
    train_file = base_path + "/train_edges.txt"
    valid_file = base_path + "/valid_edges.txt"
    test_file = base_path + "/test_edges.txt"
    if Path(train_file).exists():
        Path(train_file).unlink()
    if Path(valid_file).exists():
        Path(valid_file).unlink()
    if Path(test_file).exists():
        Path(test_file).unlink()
    files = []

    num_line_skip = (num_line_skip if num_line_skip is not None else
                     get_header_length(input_dataset, entry_regex))
    chunksize = 10 ** 7
    if validation_fraction == 0:
        with open(train_file, "a") as f, open(test_file, "a") as h:
            for chunk in pd.read_csv(
                    input_dataset, sep=delim, header=None, chunksize=chunksize,
                    skiprows=num_line_skip, usecols=data_cols, dtype=str):

                train, test = (np.split(chunk.sample(frac=1),
                               [int(train_fraction*len(chunk))]))
                train = np.asarray(train, dtype=np.str_)
                test = np.asarray(test, dtype=np.str_)

                np.savetxt(f, train, fmt='%s', delimiter=delim)
                np.savetxt(h, test, fmt='%s', delimiter=delim)
        files += [train_file, test_file]

    elif test_fraction == 0:
        with open(train_file, "a") as f, open(valid_file, "a") as g:
            for chunk in pd.read_csv(
                    input_dataset, sep=delim, header=None, chunksize=chunksize,
                    skiprows=num_line_skip, usecols=data_cols, dtype=str):

                train, valid = np.split(chunk.sample(frac=1),
                                        [int(train_fraction*len(chunk))])
                train = np.asarray(train, dtype=np.str_)
                valid = np.asarray(valid, dtype=np.str_)

                np.savetxt(f, train, fmt='%s', delimiter=delim)
                np.savetxt(g, valid, fmt='%s', delimiter=delim)
        files += [train_file, valid_file]
    else:
        with open(train_file, "a") as f,\
             open(valid_file, "a") as g,\
             open(test_file, "a") as h:
            for chunk in pd.read_csv(input_dataset, sep=delim, header=None,
                                     chunksize=chunksize,
                                     skiprows=num_line_skip, usecols=data_cols,
                                     dtype=str):
                train, valid, test = np.split(
                    chunk.sample(frac=1),
                    [int(train_fraction*len(chunk)),
                     int((train_fraction + validation_fraction)*len(chunk))])

                train = np.asarray(train, dtype=np.str_)
                valid = np.asarray(valid, dtype=np.str_)
                test = np.asarray(test, dtype=np.str_)

                np.savetxt(f, train, fmt='%s', delimiter=delim)
                np.savetxt(g, valid, fmt='%s', delimiter=delim)
                np.savetxt(h, test, fmt='%s', delimiter=delim)
        files += [train_file, valid_file, test_file]

    return files, 0, list(range(len(data_cols)))


def get_header_length(input_file, entry_regex):
    num_line_skip = 0
    with open(input_file, 'r') as f:
        n = 0
        try:
            a = next(f)
            while not re.search(entry_regex, a):
                a = next(f)
                num_line_skip += 1
                n += 1
                if n == 100:
                    raise StopIteration()
        except StopIteration:
            raise RuntimeError("Please give number of rows to skip " +
                               "at file header.")

        if a == n:
            raise RuntimeWarning("No nodes detected, dataset format may " +
                                 "be incorrect.")

    return num_line_skip


def partition_edges(edges, num_partitions, num_nodes):
    partition_size = int(np.ceil(num_nodes / num_partitions))
    src_partitions = edges[:, 0] // partition_size
    dst_partitions = edges[:, 2] // partition_size
    dst_args = np.argsort(dst_partitions, kind="stable")
    # edges = edges[dst_args]
    src_args = np.argsort(src_partitions[dst_args], kind="stable")
    edges = edges[dst_args[src_args]]
    offsets = [len(list(y)) for x, y in
               itertools.groupby(dst_partitions[dst_args[src_args]])]

    return edges, offsets


def join_files(files, regex, num_line_skip, data_cols, delim):
    assert(len(files) > 1)
    base_path = "/".join(files[0].split("/")[:-1])
    joined_file = base_path + "/joined_file.txt"
    if Path(joined_file).exists():
        Path(joined_file).unlink()

    with open(joined_file, "a") as f:
        nl = 0
        for file in files:
            num_line_skip = (num_line_skip if num_line_skip is not None
                             else get_header_length(file, regex))
            for chunk in pd.read_csv(
                    file, header=None, skiprows=num_line_skip,
                    chunksize=10 ** 7,
                    sep=delim, usecols=data_cols, dtype=str):
                np.savetxt(f, np.array(chunk, dtype=np.str_),
                           fmt="%s", delimiter=delim)
                nl += chunk.shape[0]

    return [joined_file], 0, list(range(len(data_cols)))


def general_parser(files, format, output_dir, delim="", num_partitions=1,
                   dtype=np.int32, remap_ids=True, dataset_split=(0, 0),
                   start_col=0, num_line_skip=None):
    rel_idx = format[0].find('r')
    src_idx = format[0].find('s')
    dst_idx = format[0].find('d')

    output_dir = output_dir[0].strip("/")
    output_dir = output_dir + "/"

    if rel_idx == -1:
        data_cols = [(i+start_col) for i in [0, 1]]
    else:
        data_cols = [(i+start_col) for i in [0, 1, 2]]

    if delim == "":
        with open(files[0], 'r') as input_f:
            delim = csv.Sniffer().sniff(input_f.read(10000)).delimiter
            input_f.seek(0)
        print(f"Detected delimiter: {delim}")

    if src_idx == -1 or dst_idx == -1:
        raise RuntimeError("Wrong format: source or destination not found.")
    elif rel_idx == -1:
        regex = r"^[^\s]+" + delim + r"[^\s]+$"
    else:
        regex = r"^[^\s]+" + delim + r"[^\s]+" + delim + r"[^\s]+$"

    nodes = set()
    if rel_idx != -1:
        rels = set()
    num_edges = 0
    num_edges_f = []
    num_file_read = 0

    if (len(files) > 3):
        print("Reconstructing data")
        files, num_line_skip, data_cols = join_files(
                                            files, regex, num_line_skip,
                                            data_cols, delim)

    if (len(files) == 1 and dataset_split != (0, 0)):
        print("Splitting data")
        files, num_line_skip, data_cols = split_dataset(
                                            files[0], dataset_split[0],
                                            dataset_split[1], regex,
                                            num_line_skip, data_cols, delim)

    for file in files:
        numlines = 0
        num_file_read += 1
        print(f"Reading in {file}   {num_file_read}/{len(files)}")
        num_line_skip = (num_line_skip if num_line_skip is not None
                         else get_header_length(file, regex))
        chunksize = 10 ** 7
        try:
            for chunk in pd.read_csv(file, sep=delim, header=None,
                                     chunksize=chunksize,
                                     skiprows=num_line_skip, usecols=data_cols,
                                     dtype=str):
                num_edges += chunk.shape[0]
                numlines += chunk.shape[0]
                un_set = set(np.unique(chunk[chunk.columns.values[src_idx]]))
                nodes = nodes | un_set
                un_set = set(np.unique(chunk[chunk.columns.values[dst_idx]]))
                nodes = nodes | un_set
                if rel_idx != -1:
                    un_set = (set(np.unique(
                                        chunk[chunk.columns.values[rel_idx]])))
                    rels = rels | un_set
        except pd.errors.ParserError:
            raise RuntimeError("Incorrect delimiter, must assign manually.")

        num_edges_f.append(numlines)

    node_ids = None
    if remap_ids:
        node_ids = np.random.permutation(len(nodes))
    else:
        node_ids = np.arange(len(nodes))
    nodes_dict = dict(zip(nodes, node_ids))

    if rel_idx != -1:
        rel_ids = None
        if remap_ids:
            rel_ids = np.random.permutation(len(rels))
        else:
            rel_ids = np.arange(len(rels))
        rels_dict = dict(zip(rels, rel_ids))

    print("Number of instance per file:" + str(num_edges_f))
    print("Number of nodes: " + str(len(node_ids)))
    print("Number of edges: " + str(num_edges))
    num_rels = 1 if rel_idx == -1 else len(rel_ids)
    print("Number of relations: " + str(num_rels))
    print(f"Delimiter: ~{delim}~")

    i = 0
    if len(files) == 1 or len(files) >= 3:
        temp_files = files
        if len(files) == 3:
            temp_files = [files[0]]

        train_out = output_dir + "train_edges.pt"
        with open(train_out, "wb") as f:
            for file in temp_files:
                for chunk in pd.read_csv(file, sep=delim, header=None,
                                         chunksize=chunksize,
                                         skiprows=num_line_skip,
                                         usecols=data_cols, dtype=str):

                    src_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[src_idx]]))
                    dst_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[dst_idx]]))
                    if rel_idx != -1:
                        rels = np.vectorize(rels_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[rel_idx]]))
                        edges = np.stack([src_nodes, rels, dst_nodes]).T
                    else:
                        edges = np.stack([src_nodes, np.zeros_like(src_nodes),
                                          dst_nodes]).T

                    edges = edges.astype(dtype)
                    f.write(bytes(edges))
                    i += chunksize

            if num_partitions > 1:
                f.seek(0)
                edges = np.fromfile(train_out, dtype=dtype).reshape(-1, 3)
                edges, offsets = partition_edges(edges, num_partitions,
                                                 len(node_ids))
                f.write(bytes(edges))
                with open(output_dir + "train_edges_partitions.txt", "w") as g:
                    g.writelines([str(o) + "\n" for o in offsets])

    if len(files) > 1 and len(files) < 4:
        test_out = output_dir + "test_edges.pt"
        valid_out = output_dir + "valid_edges.pt"
        with open(valid_out, "wb") as f:
            for chunk in pd.read_csv(files[1], sep=delim, header=None,
                                     chunksize=chunksize,
                                     skiprows=num_line_skip, usecols=data_cols,
                                     dtype=str):
                src_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                         chunk[chunk.columns.values[src_idx]]))
                dst_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                         chunk[chunk.columns.values[dst_idx]]))
                if rel_idx != -1:
                    rels = np.vectorize(rels_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[rel_idx]]))
                    edges = np.stack([src_nodes, rels, dst_nodes]).T
                else:
                    edges = np.stack([src_nodes, np.zeros_like(src_nodes),
                                     dst_nodes]).T

                edges = edges.astype(dtype)
                f.write(bytes(edges))
                i += chunksize
        if len(files) > 2:
            with open(test_out, "wb") as f:
                for chunk in pd.read_csv(files[2], sep=delim, header=None,
                                         chunksize=chunksize,
                                         skiprows=num_line_skip,
                                         usecols=data_cols, dtype=str):
                    src_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[src_idx]]))
                    dst_nodes = np.vectorize(nodes_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[dst_idx]]))
                    if rel_idx != -1:
                        rels = np.vectorize(rels_dict.get)(np.asarray(
                                        chunk[chunk.columns.values[rel_idx]]))
                        edges = np.stack([src_nodes, rels, dst_nodes]).T
                    else:
                        edges = np.stack([src_nodes, np.zeros_like(src_nodes),
                                         dst_nodes]).T

                    edges = edges.astype(dtype)
                    f.write(bytes(edges))
                    i += chunksize

    node_mapping = output_dir + "node_mapping.txt"
    node_mapping_tens = output_dir + "node_mapping.bin"
    with open(node_mapping, "w") as f, open(node_mapping_tens, "wb") as g:
        tens = np.array(list(nodes_dict.values()))
        g.write(bytes(tens))
        f.write("\n".join(str(key) for key in nodes_dict.keys()))

    if rel_idx != -1:
        rel_mapping = output_dir + "rel_mapping.txt"
        rel_mapping_tens = output_dir + "rel_mapping.bin"
        with open(rel_mapping, "w") as f, open(rel_mapping_tens, "wb") as g:
            tens = np.array(list(rels_dict.values()))
            g.write(bytes(tens))
            f.write("\n".join(str(key) for key in rels_dict.keys()))

    output_stats = np.zeros(3)
    output_stats[:len(num_edges_f)] = num_edges_f
    output_stats = list(output_stats)
    output_stats.insert(0, num_rels)
    output_stats.insert(0, len(node_ids))
    return output_stats


if __name__ == "__main__":
    '''
        Args: format(s,d,r) output_directory csv_file(s)
    '''
    parser = argparse.ArgumentParser(description='General CSV Converter',
                                     usage="general csv converter")
    parser.add_argument('format', type=str, nargs=1,
                        metavar="format: source(s), relation(r)," +
                        "destination(d)", help="Format of relation")
    parser.add_argument('output_directory', nargs=1,
                        metavar='output_directory', type=str,
                        help='Directory to put graph data')
    parser.add_argument("files", metavar="dataset file paths", type=str,
                        nargs='+',
                        help="path to dateset files([train, valid, test])")
    parser.add_argument('num_partitions', metavar='num_partitions',
                        type=int,
                        help='Number of partitions to split the edges into')
    args = parser.parse_args()

    general_parser(np.array(args.files).flatten(), args.format,
                   args.output_directory,
                   num_partitions=args.num_partitions)
