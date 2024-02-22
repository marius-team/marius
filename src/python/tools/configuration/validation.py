import os
from pathlib import Path

import psutil
from omegaconf import MISSING

long_dtype_list = ["long", "int64"]


def get_lines_in_file(filepath):
    return int(os.popen("wc -l {}".format(filepath)).read().lstrip().split(" ")[0])


def validate_dataset_config(output_config):
    dataset_config = output_config.storage.dataset

    if dataset_config.initialized is False:
        return

    if output_config.model.learning_task == "LINK_PREDICTION":
        num_cols = 2 if dataset_config.num_relations == 1 else 3
        edges_dtype_size = 8 if output_config.storage.edges.options.dtype in long_dtype_list else 4
        edges_path = Path(dataset_config.dataset_dir + "edges")
        train_edges_filepath = edges_path / Path("train_edges.bin")
        assert (
            os.path.getsize(train_edges_filepath) == dataset_config.num_train * num_cols * edges_dtype_size
        ), "Expected size for {} is {}, got {}".format(
            str(train_edges_filepath),
            dataset_config.num_train * num_cols * edges_dtype_size,
            os.path.getsize(train_edges_filepath),
        )

        test_edges_filepath = edges_path / Path("test_edges.bin")
        if dataset_config.num_test is not MISSING and dataset_config.num_test != -1:
            if not test_edges_filepath.exists():
                raise ValueError("{} does not exist".format(str(test_edges_filepath)))

            assert (
                os.path.getsize(test_edges_filepath) == dataset_config.num_test * num_cols * edges_dtype_size
            ), "Expected size for {} is {}, got {}".format(
                str(test_edges_filepath),
                dataset_config.num_test * num_cols * edges_dtype_size,
                os.path.getsize(test_edges_filepath),
            )

            test_edges_partitions_filepath = edges_path / Path("test_partition_offsets.txt")
            if (
                not output_config.storage.full_graph_evaluation
                and not test_edges_partitions_filepath.exists()
                and output_config.storage.embeddings.type == "PARTITION_BUFFER"
            ):
                raise ValueError(
                    "{} does not exist, required for partitioned eval".format(test_edges_partitions_filepath)
                )

        validation_edges_filepath = edges_path / Path("validation_edges.bin")
        if dataset_config.num_valid is not MISSING and dataset_config.num_valid != -1:
            if not validation_edges_filepath.exists():
                raise ValueError("{} does not exist".format(str(validation_edges_filepath)))

            assert (
                os.path.getsize(validation_edges_filepath) == dataset_config.num_valid * num_cols * edges_dtype_size
            ), "Expected size for {} is {}, got {}".format(
                str(validation_edges_filepath),
                dataset_config.num_valid * num_cols * edges_dtype_size,
                os.path.getsize(validation_edges_filepath),
            )

            valid_edges_partitions_filepath = edges_path / Path("validation_partition_offsets.txt")
            if (
                not output_config.storage.full_graph_evaluation
                and not valid_edges_partitions_filepath.exists()
                and output_config.storage.embeddings.type == "PARTITION_BUFFER"
            ):
                raise ValueError(
                    "{} does not exist, required for partitioned eval".format(valid_edges_partitions_filepath)
                )

        relation_mapping_filepath = edges_path / Path("relation_mapping.txt")
        if dataset_config.num_relations > 1:
            if not relation_mapping_filepath.exists():
                raise ValueError("{} does not exist".format(str(relation_mapping_filepath)))

            num_lines = get_lines_in_file(relation_mapping_filepath)
            if num_lines != dataset_config.num_relations:
                raise ValueError(
                    "Expected {} lines in file {}, but found {}".format(
                        dataset_config.num_relations, str(relation_mapping_filepath), num_lines
                    )
                )

    if output_config.model.learning_task == "NODE_CLASSIFICATION":
        nodes_dtype_size = 8 if output_config.storage.nodes.options.dtype in long_dtype_list else 4
        nodes_path = Path(dataset_config.dataset_dir + "nodes")
        train_nodes_filepath = nodes_path / Path("train_nodes.bin")
        if not train_nodes_filepath.exists():
            raise ValueError("{} does not exist".format(str(train_nodes_filepath)))

        assert (
            os.path.getsize(train_nodes_filepath) == dataset_config.num_train * nodes_dtype_size
        ), "Expected size for {} is {}, got {}".format(
            str(train_nodes_filepath),
            dataset_config.num_train * nodes_dtype_size,
            os.path.getsize(train_nodes_filepath),
        )

        test_nodes_filepath = nodes_path / Path("test_nodes.bin")
        if dataset_config.num_test is not MISSING and dataset_config.num_test != -1:
            if not test_nodes_filepath.exists():
                raise ValueError("{} does not exist".format(str(test_nodes_filepath)))

            assert (
                os.path.getsize(test_nodes_filepath) == dataset_config.num_test * nodes_dtype_size
            ), "Expected size for {} is {}, got {}".format(
                str(test_nodes_filepath),
                dataset_config.num_test * nodes_dtype_size,
                os.path.getsize(test_nodes_filepath),
            )

        valid_nodes_filepath = nodes_path / Path("validation_nodes.bin")
        if dataset_config.num_valid is not MISSING and dataset_config.num_valid != -1:
            if not valid_nodes_filepath.exists():
                raise ValueError("{} does not exist".format(str(valid_nodes_filepath)))

            assert (
                os.path.getsize(valid_nodes_filepath) == dataset_config.num_valid * nodes_dtype_size
            ), "Expected size for {} is {}, got {}".format(
                str(valid_nodes_filepath),
                dataset_config.num_valid * nodes_dtype_size,
                os.path.getsize(valid_nodes_filepath),
            )


def validate_storage_config(output_config):
    storage_config = output_config.storage
    dataset_config = storage_config.dataset

    if dataset_config.initialized is False:
        return

    if storage_config.embeddings.type != "PARTITION_BUFFER" and storage_config.features.type != "PARTITION_BUFFER":
        return

    edges_path = Path(dataset_config.dataset_dir + "edges")
    train_edges_partitions_filepath = edges_path / Path("train_partition_offsets.txt")
    if not train_edges_partitions_filepath.exists():
        raise ValueError(
            "{} does not exist, required for PARTITION_BUFFER mode".format(str(train_edges_partitions_filepath))
        )

    num_lines = get_lines_in_file(train_edges_partitions_filepath)
    num_partitions = storage_config.embeddings.options.num_partitions
    assert num_lines == num_partitions**2, (
        "Expected to see {} lines in {}, but found {} lines\n"
        "marius_preprocess was likely run with sqrt({}) partitions, "
        "but config file has {} partitions".format(
            num_partitions**2, str(train_edges_partitions_filepath), num_lines, num_lines, num_partitions
        )
    )

    test_edges_partitions_filepath = edges_path / Path("test_partition_offsets.txt")
    if test_edges_partitions_filepath.exists():
        num_lines = get_lines_in_file(test_edges_partitions_filepath)
        assert num_lines == num_partitions**2, (
            "Expected to see {} lines in {}, but found {} lines\n"
            "marius_preprocess was likely run with sqrt({}) partitions, "
            "but config file has {} partitions".format(
                num_partitions**2, str(test_edges_partitions_filepath), num_lines, num_lines, num_partitions
            )
        )

    valid_edges_partitions_filepath = edges_path / Path("validation_partition_offsets.txt")
    if valid_edges_partitions_filepath.exists():
        num_lines = get_lines_in_file(valid_edges_partitions_filepath)
        assert num_lines == num_partitions**2, (
            "Expected to see {} lines in {}, but found {} lines\n"
            "marius_preprocess was likely run with sqrt({}) partitions, "
            "but config file has {} partitions".format(
                num_partitions**2, str(valid_edges_partitions_filepath), num_lines, num_lines, num_partitions
            )
        )

    return


def check_encoder_layer_dimensions(output_config):
    if output_config.model.encoder is MISSING or output_config.model.encoder == -1:
        raise ValueError("No Encoder layer found. Expected to see at least 1 layer")
    embeddings_output_dim = -1
    features_output_dim = -1
    layers = output_config.model.encoder.layers
    # ensure that each layer has correct number of inputs and outputs
    for stage_idx, layer_list in enumerate(layers):
        for layer_idx, layer in enumerate(layer_list):
            if layer.type == "EMBEDDING":
                assert (
                    layer.input_dim == -1
                ), "Expected Embedding layer to have no input, but found input dim as {}".format(layer.input_dim)
                assert (
                    layer.output_dim > 0
                ), "Expected output dimension for Embedding layer to be > 0, but found {}".format(layer.output_dim)
                embeddings_output_dim = layer.output_dim if embeddings_output_dim == -1 else embeddings_output_dim
                assert (
                    embeddings_output_dim == layer.output_dim
                ), "All Embedding Layers must have the same output dimension, found {} and {}".format(
                    embeddings_output_dim, layer.output_dim
                )
                continue

            if layer.type == "FEATURE":
                assert (
                    layer.input_dim == -1
                ), "Expected Feature layer to have no input, but found input dim as {}".format(layer.input_dim)
                assert (
                    layer.output_dim > 0
                ), "Expected output dimension for Feature layer to be > 0, but found {}".format(layer.output_dim)
                features_output_dim = layer.output_dim if features_output_dim == -1 else features_output_dim
                assert (
                    features_output_dim == layer.output_dim
                ), "All Feature Layers must have the same output dimension, found {} and {}".format(
                    features_output_dim, layer.output_dim
                )
                continue

            if layer.type == "GNN":
                # should have one input and one output
                assert stage_idx > 0, "GNN Layer found in Stage 0"
                assert (
                    len(layers[stage_idx - 1]) > layer_idx
                ), "Corresponding previous Layer for GNN Layer in Stage {} not found".format(stage_idx)
                assert layers[stage_idx - 1][layer_idx].output_dim == layer.input_dim, (
                    "GNN Layer in Stage {} has input dimension of {}, "
                    "but output dimension of previous layers is {}".format(
                        stage_idx, layer.input_dim, layers[stage_idx - 1][layer_idx].output_dim
                    )
                )
                continue

            if layer.type == "REDUCTION" or layer.type == "DENSE":
                # no constraints on input and output dim
                continue

            raise ValueError("Unsupported layer type\nShould be one of EMBEDDING, FEATURE, REDUCTION, GNN, DENSE")

    # ensure that output dimension of a stage is equal to the input dimension of the next one
    for i in range(1, len(layers)):
        prev_stage_output_dim_sum = sum([layer.output_dim for layer in layers[i - 1]])
        cur_stage_input_dim_sum = sum([layer.input_dim for layer in layers[i]])
        if prev_stage_output_dim_sum != cur_stage_input_dim_sum:
            raise ValueError(
                "Encoder layers dimension mismatch.\n"
                "Output dimension of stage {} = {}\n"
                "Input dimension of stage {} = {}".format(i - 1, prev_stage_output_dim_sum, i, cur_stage_input_dim_sum)
            )


def check_gnn_layers_alignment(output_config):
    # we now know that there will be at least one layer as check_encoder_layer_dimensions was called before
    layers = output_config.model.encoder.layers
    gnn_stage_count = 0
    for i in range(len(layers)):
        for layer in layers[i]:
            if layer.type == "GNN":
                gnn_stage_count += 1
                break

    neighbor_sampling_layers = output_config.model.encoder.train_neighbor_sampling
    assert gnn_stage_count == len(
        neighbor_sampling_layers
    ), "#GNN Stages != #train_neighbor_sampling layers\nGNN Stages = {}, train_neighbor_sampling layers = {}".format(
        gnn_stage_count, len(neighbor_sampling_layers)
    )


# will remove this once AnzeXie's pr is merged
def retrieve_memory_info():
    mem = psutil.virtual_memory()
    return mem.total


# will remove this once AnzeXie's pr is merged
def get_storage_overheads(output_config):
    num_nodes = output_config.storage.dataset.num_nodes
    num_edges = output_config.storage.dataset.num_edges
    num_relations = output_config.storage.dataset.num_relations
    embedding_dim = 0 if output_config.model.encoder.embedding_dim == -1 else output_config.model.encoder.embedding_dim
    edge_dtype_size = 8 if output_config.storage.edges.options.dtype in long_dtype_list else 4
    node_dtype_size = 8 if output_config.storage.nodes.options.dtype in long_dtype_list else 4
    feature_mem_overhead = 0

    if output_config.storage.dataset.node_feature_dim != -1:
        feature_dtype_size = 8 if output_config.storage.features.options.dtype in long_dtype_list else 4
        feature_dim = output_config.storage.dataset.node_feature_dim
        feature_mem_overhead = feature_dim * num_nodes * feature_dtype_size

    node_mem_overhead = 2 * num_nodes * embedding_dim * node_dtype_size + feature_mem_overhead
    rel_mem_overhead = 2 * num_relations * embedding_dim * edge_dtype_size
    edge_mem_overhead = (
        num_edges * 2 * edge_dtype_size * 2 if num_relations == 1 else num_edges * 3 * edge_dtype_size * 2
    )

    return node_mem_overhead, rel_mem_overhead, edge_mem_overhead


def check_full_graph_evaluation(output_config):
    if output_config.storage.dataset.initialized is False:
        return

    full_graph_evaluation = output_config.storage.full_graph_evaluation
    if not full_graph_evaluation:
        return

    # replace these function call
    mem_available = retrieve_memory_info()
    node_mem_overhead, rel_mem_overhead, _ = get_storage_overheads(output_config)
    if node_mem_overhead + rel_mem_overhead > mem_available:
        raise ValueError(
            "full_graph_evaluation set to true, but not enough memory available for storing node and relation"
            " embeddings\nRequired memory = {} bytes, Available memory = {} bytes".format(
                str(node_mem_overhead + rel_mem_overhead), str(mem_available)
            )
        )
