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
    train_file = base_path + "/splitted_train_edges.txt"
    valid_file = base_path + "/splitted_valid_edges.txt"
    test_file = base_path + "/splitted_test_edges.txt"
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


def check_given_num_line_skip_start_col(input_file, num_line_skip, data_cols, 
        delim, start_col):
    with open(input_file, 'r') as f:
        for i in range(num_line_skip):
            line = next(f)

        line = next(f)
        splitted_line = line.split(delim)
        if len(splitted_line) - start_col < len(data_cols):
            return False
    
    return True


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
    assert(len(files) != 0), "Number of data files cannot be 0"
    assert(len(format) == 1), "Format is specified incorrectly"
    assert((start_col == 0) or (start_col != 0 and num_line_skip != None)), \
                "Need to specify num_line_skip if start_col is specified"
    assert(num_partitions > 0)

    rel_idx = format[0].find('r')
    src_idx = format[0].find('s')
    dst_idx = format[0].find('d')
    assert((len(format[0]) == 3 and rel_idx != -1 and
            src_idx != -1 and dst_idx != -1) or
           (len(format[0]) == 2 and dst_idx != -1 and
            src_idx != -1)), "Format is specified incorrectly"

    assert(Path(output_dir[0]).exists()), "Output directory not found"
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
        print(f"Detected delimiter: ~{delim}~")
    if num_line_skip is not None:
        assert(check_given_num_line_skip_start_col(files[0], num_line_skip,
               data_cols, delim, start_col)), "Incorrect num_line_skip given"

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

    output_stats = np.zeros(3, dtype=int)
    output_stats[:len(num_edges_f)] = num_edges_f
    output_stats = list(output_stats)
    output_stats.insert(0, num_rels)
    output_stats.insert(0, len(node_ids))
    return output_stats


def set_args():
    parser = argparse.ArgumentParser(
        description='csv converter', prog='csv_converter',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('files', metavar='files', nargs='+', type=str,
                        help='Data files')
    parser.add_argument('format', metavar='format', nargs=1, type=str,
                        help='Format of data, eg. srd')
    parser.add_argument('output_dir', metavar='output_dir', nargs=1, type=str,
                        help='Output directory for preprocessed data')
    parser.add_argument('--delim', '-d', metavar='delim', type=str,
                        default="",
                        help='Specifies the delimiter')
    parser.add_argument('--num_partitions', '-np', metavar='num_partitions',
                        type=int, default=1, help='number of partitions')
    parser.add_argument('--dtype', metavar='dtype', type=np.dtype,
                        default=np.int32,
                        help='Indicates the numpy.dtype')
    parser.add_argument('--not_remap_ids', action='store_false',
                        help='If set, will not remap ids')
    parser.add_argument('--dataset_split', '-ds', metavar='dataset_split',
                        nargs=2, type=float, default=[0, 0],
                        help='Split dataset into specified fractions')
    parser.add_argument('--start_col', '-sc', metavar='start_col', type=int,
                        default=0,
                        help='Indicates the column index to start from')
    parser.add_argument('--num_line_skip', '-nls', metavar='num_line_skip',
                        type=int, default=None,
                        help='Indicates number of lines to ' +
                             'skip from the beginning')

    args = parser.parse_args()
    arg_dict = vars(args)
    if arg_dict.get("dataset_split") is not None:
        arg_dict.update({"dataset_split": tuple(arg_dict.get("dataset_split"))})
    
    #arg_dict.update({"dtype": np.dtype(arg_dict.get("dtype"))}) # FIXME: argparse not taking np.int32
    return arg_dict


def main():
    arg_dict = set_args()
    general_parser(arg_dict.get("files"), arg_dict.get("format"),
                   arg_dict.get("output_dir"), arg_dict.get("delim"),
                   arg_dict.get("num_partitions"), arg_dict.get("dtype"),
                   arg_dict.get("not_remap_ids"), arg_dict.get("dataset_split"),
                   arg_dict.get("start_col"), arg_dict.get("num_line_skip"))

if __name__ == "__main__":
    main()
