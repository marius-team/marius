"""Converter for CSV, TSV and TXT dataset files.

This module contains the functions for converting CSV, TSV and TXT dataset
files into Marius input formats.
"""

import argparse
import re
from pathlib import Path
import csv
import pandas as pd
import numpy as np
import itertools


def split_dataset(input_dataset, validation_fraction, test_fraction,
                  entry_regex, num_line_skip, data_cols,
                  delim):
    """Splits dataset into training, validation and testing sets.

    Splits one input dataset file into training, validation and testing sets
    according to the given fractions. During the splitting process, all edges
    in the input dataset are randomly sampled into training set, validation
    set and testing set according to validation_fraction and test_fraction.
    Then only these edges are written to splitted_train_edges.txt, 
    splitted_valid_edges.txt and splitted_test_edges.txt files in the same
    directory of the input dataset file. If either of validation_fraction or
    test_fraction is set to zero, the corresponding file will not be created.
    The following files are created by this function:
        splitted_train_edges.txt: File containing training set edges.
        splitted_valid_edges.txt: File containing validation set edges.
        splitted_test_edges.txt: File containing testing set edges.
    
    Args:
        input_dataset: The path to the original data file to be splitted.
        validation_fraction: The proportion of the input dataset that will be
            put into the validation set.
        test_fraction: The proportion of the input dataset that will be put
            into the testing set.
        entry_regex: The regular expression of the representation of an edge in
            the dataset.
        num_line_skip: Number of lines to skip as the header of the dataset
            file.
        data_cols: A list of index indicating which columns in the dataset file
            compose the edges.
        delim: The delimiter between two columns in the dataset file.

    Returns:
        The list of file path to splitted_train_edges.txt,
        splitted_valid_edges.txt and splitted_test_edges.txt are returned. In
        the meantime, the num_line_skip is set to 0 and data_cols is set to
        the first two or three columns based on whether there is relation in
        the dataset for the downstream preprocessing operations.
    """
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
    """Detects the number of rows to skip as the file header.

    This function counts the number of rows do not contain the substring that
    matches the edge regular expression from the start of the file to detects
    the number of rows to skip as the file header.
    
    Args:
        input_file: The object file for detecting number of header rows.
        entry_regex: The regular expression of the representation of an edge in
            the dataset.
    
    Returns:
        The number of rows to skip as the file header.

    Raises:
        RuntimeError: An error occurred when the process of detecting file
            header length fails. A common failure case is that the file header
            also contains the regular expression for edges. In this case,
            number of rows to skip as file header should be manually set.
    """
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

    return num_line_skip


def check_given_num_line_skip_start_col(input_file, num_line_skip, data_cols,
                                        delim, start_col):
    """Check if the given combination of num_line_skip and start_col is valid.

    This function splits the first row after the file header with the given
    delimiter and check if start_col index is within the valid range (less than
    the number of tokens splitted).

    Args:
        input_file: A dataset file used to check the validity of the given
            combination of num_line_skip and start_col.
        num_line_skip: Number of lines to skip as the header of the dataset
            file.
        data_cols: A list of index indicating which columns in the dataset file
            compose the edges.
        delim: The delimiter between two columns in the dataset file.
        start_col: The index of the first column of the edge representations in
            the dataset file.
    
    Returns:
        True if the given combination of num_line_skip and start_col is valid.
        False if the given combination of num_line_skip and start_col is not 
        valid.
    """
    with open(input_file, 'r') as f:
        for i in range(num_line_skip):
            line = next(f)

        line = next(f)
        splitted_line = line.split(delim)
        if len(splitted_line) - start_col < len(data_cols):
            return False

    return True


def partition_edges(edges, num_partitions, num_nodes):
    """Split the nodes into num_partitions partitions.

    In the case of large scale graphs that have an embedding table which
    exceeds CPU memory capacity, this function can partition the graph nodes
    uniformly into num_partitions partitions and group the edges into edge
    buckets. This partitioning method assumes that all edges fit in memory.
    See partition_scheme for more details.

    Args:
        edges: All edges of original dataset.
        num_partitions: The number of graph partitions that the graph nodes are
            uniformly partitioned into.
        num_nodes: The total number of nodes.

    Returns:
        Reordered edges and a list of offsets indicating node partitions.
    """
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
    """Joins multiple dataset files into one dataset file

    Joins edges from multiple dataset files into one dataset file. This
    function should only be called when there are more than one file. During
    the process of joining, only edges from each dataset file is extracted and
    then written to joined_file.txt.
    The following files are created by this function:
        joined_file.txt: The file contains all edges from the current dataset.

    Args:
        files: A list of dataset files to be joined.
        regex: The regular expression of the representation of an edge in the
            dataset.
        num_line_skip: Number of lines to skip as the header of the dataset
            file.
        data_cols: A list of index indicating which columns in the dataset file
            compose the edges.
        delim: The delimiter between two columns in the dataset file.

    Returns:
        The joint file is returned as a list of one file. Meaning while, 
        num_line_skip is set to zero and data_cols is set to the first two or
        three columns depends on if the edges in the dataset has relations.
    """
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
                   dtype=np.int32, remap_ids=True, dataset_split=(-1, -1),
                   start_col=0, num_line_skip=None):
    """Parses dataset in the format of CSV, TSV and TXT to marius input format.

    This function retrieves all edges from given dataset file. Each node and
    edge_type is randomly assigned an integer id. The mappings from these
    integer ids to the original ids are stored in node_mapping.txt and
    rel_mapping.txt.
    The original edges list is converted to an [|E|, 3] int32 tensor, shuffled and
    then the contents of the tensor are written to the train_edges.pt file
    and/or valid_edges.pt and test_edges.pt depend on dataset_split.
    The following files are created by this function:
        train_edges.pt: Dump of tensor memory for edges in the training set.
        valid_edges.pt: Dump of tensor memroy for edges in the validation set.
        test_edges.pt: Dump of tensor memroy for edges in the testing set.
        node_mapping.txt: Mapping of original node ids to unique int32 ids.
        rel_mapping.txt: Mapping of original edge_type ids to unique int32 ids.

        If num_partitions is set to a value greater than one, then the
        following file is also created:
        train_edges_partitions.txt: text file with num_partitions^2 lines,
            where each line denotes the size of an edge bucket

    Args:
        files: The list of original dataset files. If there are three files,
            they are treated as training, validation and testing set based on
            their order by default (if dataset_split is not set).
        format: A string denotes the order of edge components. The value of
            this string can only be "s" for source nodes, "r" for relation,
            "d" for destination nodes. The length of this string can be two or
            three depends on if the edges have relations.
        output_dir: The directory where all the files created will be stored.
        delim: The delimiter between two columns in the dataset file.
        num_partitions: The number of graph partitions that the graph nodes are
            uniformly partitioned into.
        dtype: The data type of the edge list. The common values for this
            argument is np.int32 or np.int64. If there are less then 2 billion
            nodes (which is almost every dataset), int32 should be used. If the
            value is set to np.int32, then each edge takes 3*4/2*4 bytes of
            space to store. In the case of np.int64, each edge takes 3*8/2*8
            bytes of space to store.
        remap_ids: Whether to assign node and relations random ids or
            sequential ids based on their order in original dataset file.
        dataset_split: The proportion of the input data that will be used for
            validation and testing during training. The argument takes a tuple
            of length two where the first value is the proportion of validation
            set and the second value is the proportion of testing set.
        start_col: The index of the first column of the edge representations in
            the dataset file.
        num_line_skip: Number of lines to skip as the header of the dataset
            file.

    Returns:
        The created files described above will be stored into the output_dir
        directory. Statistics of the preprocessed dataset are put into a list
        and returned. These statistics are placed in the following order:
        number of edges in train_edges.pt, number of edges in valid_edges.pt,
        number of edges in test_edgs.pt, number of relations, and number of
        nodes. These statistics are also printed to the terminal.

    Raises:
        RuntimeError: An error occurred when the denotation of source node "s"
            or destination node "d" is not found in the value of argument
            format.
            This error also occurred if the delimiter given or the delimiter
            detected is not correct. In this case, a new delimiter should be
            assigned manually.
            Detailed helper messages indicating the possible causes are printed
            when this error is raised.
    """
    assert(len(files) != 0), "Number of data files cannot be 0"
    assert(len(format) == 1), "Format is specified incorrectly"
    assert((start_col == 0) or
           (start_col != 0 and num_line_skip is not None)),\
        "Need to specify num_line_skip if start_col is specified"
    assert(num_partitions > 0)

    rel_idx = format[0].find('r')
    src_idx = format[0].find('s')
    dst_idx = format[0].find('d')
    assert((len(format[0]) == 3 and rel_idx != -1 and
            src_idx != -1 and dst_idx != -1) or
           (len(format[0]) == 2 and dst_idx != -1 and
            src_idx != -1)), "Format is specified incorrectly"

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)
    assert(Path(output_dir).exists()), "Output directory not found"
    output_dir = output_dir.strip("/")
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
        print("Merging data")
        files, num_line_skip, data_cols = join_files(
                                            files, regex, num_line_skip,
                                            data_cols, delim)

    if (len(files) == 1 and dataset_split != (0, 0) and dataset_split != (-1, -1)):
        print("Splitting data")
        files, num_line_skip, data_cols = split_dataset(
                                            files[0], dataset_split[0],
                                            dataset_split[1], regex,
                                            num_line_skip, data_cols, delim)

    if (len(files) > 1 and dataset_split != (-1, -1)):
        print("Merging data")
        files, num_line_skip, data_cols = join_files(
                                            files, regex, num_line_skip,
                                            data_cols, delim)
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
    node_dict_items = np.array(list(nodes_dict.items()))
    np.savetxt(node_mapping, node_dict_items, fmt='%s', delimiter='\t')

    if rel_idx != -1:
        rel_mapping = output_dir + "rel_mapping.txt"
        rels_dict_items = np.array(list(rels_dict.items()))
        np.savetxt(rel_mapping, rels_dict_items, fmt='%s', delimiter='\t')

    output_stats = np.zeros(3, dtype=int)
    output_stats[:len(num_edges_f)] = num_edges_f
    output_stats = list(output_stats)
    output_stats.insert(0, num_rels)
    output_stats.insert(0, len(node_ids))
    return output_stats


def set_args():
    """Sets command line arguments for this csv_converter modules.

    Returns:
        A dict containing all command line arguments and their values.
    """
    parser = argparse.ArgumentParser(
        description='csv converter', prog='csv_converter',
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('files', metavar='files', nargs='+', type=str,
                        help='Data files')
    parser.add_argument('format', metavar='format', nargs=1, type=str,
                        help='Format of data, eg. srd')
    parser.add_argument('output_dir', metavar='output_dir', type=str,
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
                        nargs=2, type=float, default=[-1, -1],
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
