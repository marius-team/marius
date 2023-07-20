import argparse
import os
import pathlib
from argparse import RawDescriptionHelpFormatter

import numpy as np
import pandas as pd

import marius as m
from marius.tools.configuration.constants import PathConstants
from marius.tools.prediction.link_prediction import infer_lp
from marius.tools.prediction.node_classification import infer_nc
from marius.tools.preprocess.converters.partitioners.torch_partitioner import partition_edges
from marius.tools.preprocess.converters.readers.pandas_readers import PandasDelimitedFileReader
from marius.tools.preprocess.converters.torch_converter import (
    SUPPORTED_DELIM_FORMATS,
    apply_mapping1d,
    apply_mapping_edges,
    dataframe_to_tensor,
)

import torch  # isort:skip


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def set_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tool for performing link prediction or node classification inference with trained models.\n\nLink"
            " prediction example usage: \nmarius_predict <trained_config> --output_dir results/ --metrics mrr mean_rank"
            " hits1 hits10 hits50 --save_scores --save_ranks \nAssuming <trained_config> contains a link prediction"
            " model, this command will perform link prediction evaluation over the test set of edges provided in the"
            " config file. Metrics are saved to results/metrics.txt and scores and ranks for each test edge are saved"
            " to results/scores.csv \n\nNode classification example usage: \nmarius_predict <trained_config>"
            " --output_dir results/ --metrics accuracy --save_labels \nThis command will perform node classification"
            " evaluation over the test set of nodes provided in the config file. Metrics are saved to"
            " results/metrics.txt and labels for each test node are saved to results/labels.csv \n\nCustom inputs:"
            " \nThe test set can be directly specified setting --input_file <test_set_file>. If the test set has not"
            " been preprocessed, then --preprocess_input should be enabled. The default format is a binary file, but"
            " additional formats can be specified with --input_format."
        ),
        prog="predict",
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", metavar="config", required=True, type=str, help="Configuration file for trained model"
    )

    parser.add_argument("--output_dir", metavar="output_dir", type=str, default="", help="Path to output directory")

    parser.add_argument(
        "--metrics", metavar="metrics", type=str, nargs="*", default=[], help="List of metrics to report"
    )

    parser.add_argument(
        "--save_labels",
        action="store_true",
        default=False,
        help=(
            "(Node Classification) If true, the node classification labels of each test node will be saved to"
            " <output_dir>/labels.csv"
        ),
    )

    parser.add_argument(
        "--save_scores",
        action="store_true",
        default=False,
        help=(
            "(Link Prediction) If true, the link prediction scores of each test edge will be saved to"
            " <output_dir>/scores.csv"
        ),
    )

    parser.add_argument(
        "--save_ranks",
        action="store_true",
        default=False,
        help=(
            "(Link Prediction) If true, the link prediction ranks of each test edge will be saved to"
            " <output_dir>/scores.csv"
        ),
    )

    parser.add_argument(
        "--batch_size", metavar="batch_size", type=int, default=10000, help="Number of examples to evaluate at a time."
    )

    parser.add_argument(
        "--num_nbrs",
        metavar="num_nbrs",
        type=list,
        default=None,
        help=(
            "Number of neighbors to sample for each GNN layer.If not provided, then the module will check if the output"
            " of the encoder has been saved after training (see storage.export_encoded_nodes). If the encoder outputs"
            " exist, the the module will skip the encode step (incl. neighbor sampling) and only perform the decode"
            " over the saved inputs.If encoder outputs are not saved, model.encoder.eval_neighbor_sampling will be used"
            " for the neighbor sampling configuration.If model.encoder.eval_neighbor_sampling does not exist, then"
            " model.encoder.train_neighbor_sampling will be used.If none of the above are given, then the model is"
            " assumed to not require neighbor sampling."
        ),
    )

    parser.add_argument(
        "--num_negs",
        metavar="num_negs",
        type=int,
        default=None,
        help=(
            "(Link Prediction) Number of negatives to compare per positive edge for link prediction. If -1, then all"
            " nodes are used as negatives. Otherwise, num_neg*num_chunks nodes will be sampled and used as negatives.If"
            " not provided, the evaluation.negative_sampling configuration will be used.if evaluation.negative_sampling"
            " is not provided, then negative sampling will not occur and only the scores for the input edges will be"
            " computed, this means that any ranking metrics cannot be calculated."
        ),
    )

    parser.add_argument(
        "--num_chunks",
        metavar="num_chunks",
        type=int,
        default=1,
        help=(
            "(Link Prediction) Specifies the amount of reuse of negative samples. "
            "A given set of num_neg sampled nodes will be reused to corrupt (batch_size // num_chunks) edges."
        ),
    )

    parser.add_argument(
        "--deg_frac",
        metavar="deg_frac",
        type=float,
        default=0.0,
        help=(
            "(Link Prediction) Specifies the fraction of the num_neg nodes sampled as negatives that should be sampled"
            " according to their degree. This sampling procedure approximates degree based sampling by sampling nodes"
            " that appear in the current batch of edges."
        ),
    )

    parser.add_argument(
        "--filtered",
        metavar="filtered",
        type=str2bool,
        default=True,
        help=(
            "(Link Prediction) If true, then false negative samples will be filtered out. "
            "This is only supported when evaluating with all nodes."
        ),
    )

    parser.add_argument(
        "--input_file",
        metavar="input_file",
        type=str,
        default="",
        help=(
            "Path to input file containing the test set, "
            "if not provided then the test set described in the configuration file will be used."
        ),
    )

    parser.add_argument(
        "--input_format",
        metavar="input_format",
        type=str,
        default="binary",
        help=(
            "Format of the input file to test. "
            "Options are [BINARY, CSV, TSV, DELIMITED] files. If DELIMITED, then --delim must be specified."
        ),
    )

    parser.add_argument(
        "--preprocess_input",
        metavar="preprocess_input",
        type=str2bool,
        default=False,
        help="If true, the input file (if provided) will be preprocessed before evaluation.",
    )

    parser.add_argument(
        "--columns",
        metavar="columns",
        type=list,
        default=[],
        help=(
            "List of column ids of input delimited file which denote the src node, edge-type, and dst node of"
            " edges.E.g. columns=[0, 2, 1] means that the source nodes are found in the first column of the file, the"
            " edge-types are found in the third column, and the destination nodes are found in the second column.For"
            " graphs without edge types, only the location node columns need to be provided. E.g. [0, 1]If the input"
            " file contains node ids rather than edges, then only a single id is needed. E.g. [2]"
        ),
    )

    parser.add_argument(
        "--header_length",
        metavar="header_length",
        type=int,
        default=0,
        help="Length of the header for input delimited file",
    )

    parser.add_argument("--delim", metavar="delim", type=str, default=None, help="Delimiter for input file")

    parser.add_argument(
        "--dtype",
        metavar="dtype",
        type=str,
        default="",
        help="Datatype of input file elements. Defaults to the dataset specified in the configuration file.",
    )

    return parser


def get_metrics(config, args):
    metrics = []
    if config.model.learning_task == m.config.LearningTask.LINK_PREDICTION:
        # setup metrics
        for metric in args.metrics:
            metric = metric.upper()

            if metric == "MRR" or metric == "MEANRECIPROCALRANK" or metric == "MEAN_RECIPROCAL_RANK":
                metrics.append(m.report.MeanReciprocalRank())
            elif metric == "MR" or metric == "MEANRANK" or metric == "MEAN_RANK":
                metrics.append(m.report.MeanRank())
            elif metric.startswith("HITS"):
                str_offset = 4
                if metric.startswith("HITS@K"):
                    str_offset = 6
                try:
                    k = int(metric[str_offset:])
                    metrics.append(m.report.Hitsk(k))
                except RuntimeError as err:
                    raise RuntimeWarning(
                        "Unable to parse k value for hits@k metric: " + metric + "\nError: " + err.__str__()
                    )

            else:
                raise RuntimeWarning("Unsupported metric for link prediction: " + metric)

    elif config.model.learning_task == m.config.LearningTask.NODE_CLASSIFICATION:
        for metric in args.metrics:
            metric = metric.upper()

            if (
                metric == "ACC"
                or metric == "ACCURACY"
                or metric == "CATEGORICAL_ACCURACY"
                or metric == "CATEGORICALACCURACY"
            ):
                metrics.append(m.report.CategoricalAccuracy())
            else:
                raise RuntimeWarning("Unsupported metric for node classification: " + metric)

    else:
        raise RuntimeError("Unsupported learning task.")

    return metrics


def get_dtype(storage_backend, args):
    str_dtype = args.dtype.lower()
    if str_dtype == "":
        if storage_backend.dtype == torch.int32:
            numpy_dtype = np.int32
            str_dtype = "int32"
        else:
            numpy_dtype = np.int64
            str_dtype = "int64"
    else:
        if str_dtype == "int32" or str_dtype == "int":
            numpy_dtype = np.int32
        elif str_dtype == "int64" or str_dtype == "long":
            numpy_dtype = np.int64
        else:
            raise RuntimeError("Unsupported datatype for input file.")

    return str_dtype, numpy_dtype


def get_columns(config, args):
    is_edges = config.model.learning_task == m.config.LearningTask.LINK_PREDICTION

    columns = args.columns
    if len(columns) == 0:
        if is_edges:
            if config.storage.dataset.num_relations > 1:
                columns = [0, 1, 2]
            else:
                columns = [0, 1]
        else:
            columns = [0]
    else:
        if is_edges:
            if config.storage.dataset.num_relations > 1:
                assert len(columns) == 3
            else:
                assert len(columns) == 2
        else:
            assert len(columns) == 1
    return columns


def infer_input_shape(config, args):
    is_edges = config.model.learning_task == m.config.LearningTask.LINK_PREDICTION

    if args.input_format.upper() == "BINARY" or args.input_format.upper() == "BIN":
        if is_edges:
            storage_backend = config.storage.edges

            file_size = os.stat(args.input_file).st_size
            _, numpy_dtype = get_dtype(storage_backend, args)

            if config.storage.dataset.num_relations > 1:
                shape = [file_size // (numpy_dtype.itemsize * 3), 3]
            else:
                shape = [file_size // (numpy_dtype.itemsize * 2), 2]

            assert shape[0] * shape[1] * numpy_dtype.itemsize == file_size
        else:
            storage_backend = config.storage.nodes

            file_size = os.stat(args.input_file).st_size
            _, numpy_dtype = get_dtype(storage_backend, args)
            shape = [file_size // numpy_dtype.itemsize]
            assert shape[0] * numpy_dtype.itemsize == file_size

    elif args.input_format.upper() in SUPPORTED_DELIM_FORMATS:
        line_count = None
        with open(args.input_format) as f:
            line_count = sum(1 for _ in f)

        if is_edges:
            if config.storage.dataset.num_relations > 1:
                shape = [line_count, 3]
            else:
                shape = [line_count, 2]
        else:
            if config.storage.dataset.num_relations > 1:
                shape = [line_count, 3]
            else:
                shape = [line_count, 2]
    else:
        raise RuntimeError("Unsupported input format. ")

    return shape


def get_nbrs_config(config, args):
    nbrs = args.num_nbrs
    if nbrs is None:
        if config.storage.export_encoded_nodes and config.model.learning_task == m.config.LearningTask.LINK_PREDICTION:
            return None

        nbrs = []
        if config.model.encoder.eval_neighbor_sampling is not None:
            for layer in config.model.encoder.eval_neighbor_sampling:
                if layer.type == m.config.NeighborSamplingLayer.ALL:
                    nbrs.append(-1)
                else:
                    nbrs.append(layer.options.num_neighbors)

            return nbrs

        if config.model.encoder.train_neighbor_sampling is not None:
            for layer in config.model.encoder.train_neighbor_sampling:
                if layer.type == m.config.NeighborSamplingLayer.ALL:
                    nbrs.append(-1)
                else:
                    nbrs.append(layer.options.num_neighbors)

            return nbrs

    return nbrs


def get_neg_config(config, args):
    if args.num_negs is None:
        num_negs = config.evaluation.negative_sampling.negatives_per_positive
        num_chunks = config.evaluation.negative_sampling.num_chunks
        deg_frac = config.evaluation.negative_sampling.degree_fraction
        filtered = config.evaluation.negative_sampling.filtered
        return num_negs, num_chunks, deg_frac, filtered
    else:
        return args.num_negs, args.num_chunks, args.deg_frac, args.filtered


def preprocess_input_file(config, args):
    assert args.preprocess_input
    assert pathlib.Path(args.input_file).exists()

    is_edges = config.model.learning_task == m.config.LearningTask.LINK_PREDICTION

    if is_edges:
        storage_backend = config.storage.edges
    else:
        storage_backend = config.storage.nodes

    shape = infer_input_shape(config, args)
    str_dtype, numpy_dtype = get_dtype(storage_backend, args)

    node_mapping_file = config.storage.dataset.dataset_dir + PathConstants.node_mapping_path
    rel_mapping_file = config.storage.dataset.dataset_dir + PathConstants.relation_mapping_path

    node_mapping_df = None
    rel_mapping_df = None

    if pathlib.Path(node_mapping_file).exists():
        node_mapping_df = pd.read_csv(node_mapping_file, sep=",", header=None)

    if pathlib.Path(rel_mapping_file).exists():
        rel_mapping_df = pd.read_csv(rel_mapping_file, sep=",", header=None)

    if args.input_format.upper() == "BINARY" or args.input_format.upper() == "BIN":
        input_tensor = torch.from_file(np.fromfile(args.filename, numpy_dtype)).resize(shape)

        if node_mapping_df is not None:
            if len(input_tensor.shape) == 2:
                input_tensor = apply_mapping_edges(input_tensor, node_mapping_df, rel_mapping_df)
            else:
                input_tensor = apply_mapping1d(input_tensor, node_mapping_df)
    else:
        columns = get_columns(config, args)

        delim = args.delim

        if delim is None:
            if args.input_format.upper() == "CSV":
                delim = ","
            elif args.input_format.upper() == "TSV":
                delim = "\t"
            else:
                raise RuntimeError("Delimiter must be specified.")

        reader = PandasDelimitedFileReader(
            args.input_file, columns=columns, header_length=args.header_length, delim=delim, dtype=str_dtype
        )

        input_df, _, _ = reader.read()

        if node_mapping_df is not None:
            if len(input_df.shape) == 2:
                input_df = apply_mapping_edges(input_df, node_mapping_df, rel_mapping_df)
            else:
                input_df = apply_mapping1d(input_df, node_mapping_df)

        input_tensor = dataframe_to_tensor(input_df)

    # TODO probably not a great way to name the preprocessed file
    input_file = "preproc_" + args.input_file.split(".")[-2] + ".bin"
    input_file_offsets = None

    num_partitions = 1
    if (
        config.storage.embeddings is not None
        and config.storage.embeddings.type == m.config.StorageBackend.PARTITION_BUFFER
    ):
        num_partitions = config.storage.embeddings.options.num_partitions
    elif (
        config.storage.features is not None and config.storage.features.type == m.config.StorageBackend.PARTITION_BUFFER
    ):
        num_partitions = config.storage.features.options.num_partitions

    if num_partitions > 1 and len(input_tensor.shape) == 2:
        input_file_offsets = args.input_file.split(".")[-2] + "_offsets.txt"
        input_tensor, offsets = partition_edges(input_tensor, config.storage.dataset.num_nodes, num_partitions)

        with open(config.storage.dataset.dataset_dir + input_file_offsets, "w") as f:
            f.writelines([str(o) + "\n" for o in offsets])

    with open(config.storage.dataset.dataset_dir + input_file, "wb") as f:
        f.write(bytes(input_tensor.numpy()))

    return input_file, input_file_offsets, storage_backend, shape


def get_input_file_storage(config, args):
    assert pathlib.Path(args.input_file).exists()

    if args.preprocess_input:
        input_file, input_file_offsets, storage_backend, shape = preprocess_input_file(config, args)
    else:
        input_file = args.input_file
        input_file_offsets = None

        is_edges = config.model.learning_task == m.config.LearningTask.LINK_PREDICTION
        if is_edges:
            storage_backend = config.storage.edges
        else:
            storage_backend = config.storage.nodes

        shape = infer_input_shape(config, args)

    if storage_backend.type is m.config.StorageBackend.DEVICE_MEMORY:
        input_storage = m.storage.InMemory(input_file, shape, storage_backend.dtype, config.storage.device)
    elif storage_backend.type is m.config.StorageBackend.HOST_MEMORY:
        input_storage = m.storage.InMemory(input_file, shape, storage_backend.dtype, torch.device("cpu"))
    elif storage_backend.type is m.config.StorageBackend.FLAT_FILE:
        input_storage = m.storage.FlatFile(input_file, shape, storage_backend.dtype)
    else:
        raise RuntimeError("Unexpected storage backend for input_file.")

    if input_file_offsets is not None:
        input_storage.read_edge_bucket_sizes(input_file_offsets)


def run_predict(args):
    config = m.config.loadConfig(args.config)
    metrics = get_metrics(config, args)

    model_dir_path = pathlib.Path(config.storage.model_dir)
    if not model_dir_path.exists():
        raise RuntimeError("Path {} with model params doesn't exist.".format(str(model_dir_path)))

    model: m.nn.Model = m.storage.load_model(args.config, train=False)
    graph_storage: m.storage.GraphModelStorage = m.storage.load_storage(args.config, train=False)

    if args.input_file != "":
        input_storage = get_input_file_storage(config, args)

        if config.model.learning_task == m.config.LearningTask.LINK_PREDICTION:
            graph_storage.storage_ptrs.edges = input_storage
        elif config.model.learning_task == m.config.LearningTask.NODE_CLASSIFICATION:
            graph_storage.storage_ptrs.nodes = input_storage
        else:
            raise RuntimeError("Unsupported learning task for inference.")
    else:
        graph_storage.setTestSet()

    output_dir = args.output_dir
    if output_dir == "":
        output_dir = config.storage.model_dir

    nbrs = get_nbrs_config(config, args)

    if config.model.learning_task == m.config.LearningTask.LINK_PREDICTION:
        num_negs, num_chunks, deg_frac, filtered = get_neg_config(config, args)
        infer_lp(
            model=model,
            graph_storage=graph_storage,
            output_dir=output_dir,
            metrics=metrics,
            save_scores=args.save_scores,
            save_ranks=args.save_ranks,
            batch_size=args.batch_size,
            num_nbrs=nbrs,
            num_negs=num_negs,
            num_chunks=num_chunks,
            deg_frac=deg_frac,
            filtered=filtered,
        )

    elif config.model.learning_task == m.config.LearningTask.NODE_CLASSIFICATION:
        infer_nc(
            model=model,
            graph_storage=graph_storage,
            output_dir=output_dir,
            metrics=metrics,
            save_labels=args.save_labels,
            batch_size=args.batch_size,
            num_nbrs=nbrs,
        )
    else:
        raise RuntimeError("Unsupported learning task for inference.")

    print("Results output to: {}".format(output_dir))


def main():
    parser = set_args()
    args = parser.parse_args()
    run_predict(args)


if __name__ == "__main__":
    main()
