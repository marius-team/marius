import argparse
import os
from pathlib import Path

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONFIG_FILE = os.path.join(HERE, "config_templates", "default_configs.txt")
DATASET_STATS = os.path.join(HERE, "dataset_stats", "dataset_stats.tsv")


def output_config(config_dict, output_dir):
    device = config_dict.get("device")
    if config_dict.get("dataset") is None:
        ds_name = "custom"
    else:
        ds_name = config_dict.get("dataset")

    file = Path(output_dir) / Path(str(ds_name) + "_" + device.lower() + ".ini")
    all_sections = [
        "general",
        "model",
        "storage",
        "training_sampling",
        "training",
        "training_pipeline",
        "evaluation",
        "evaluation_pipeline",
        "path",
        "reporting",
    ]
    opts = list(config_dict.keys())
    section_to_print = []

    for sec in all_sections:
        for key in opts:
            if key.split(".")[0] == sec:
                if sec not in section_to_print:
                    section_to_print.append(sec)

    with open(file, "w+") as f:
        for sec in section_to_print:
            f.write("[" + sec + "]\n")
            for key in opts:
                if key.split(".")[0] == sec:
                    f.write(key.split(".")[1] + "=" + str(config_dict.get(key)) + "\n")
            f.write("\n")


def read_template(file):
    with open(file, "r") as f:
        lines = f.readlines()

    keys = []
    values = []
    valid_dict = {}
    for line in lines:
        line = line.split("=")
        line[1] = line[1].rstrip()
        keys.append(line[0])
        sub_line = line[1].split("*")
        values.append(sub_line[0])
        if len(sub_line) > 1:
            valid_dict.update({line[0]: sub_line[1:]})
    config_dict = dict(zip(keys, values))

    return config_dict, valid_dict


def set_up_files(output_directory):
    try:
        if not Path(output_directory).exists():
            Path(output_directory).mkdir(parents=False, exist_ok=False)
    except FileExistsError:
        print("Directory already exists.")
    except FileNotFoundError:
        print("Incorrect parent path given for output directory.")


def update_dataset_stats(dataset, arg_dict, config_dict):
    datasets_stats = pd.read_csv(DATASET_STATS, sep="\t")
    stats_row = datasets_stats[datasets_stats["dataset"] == dataset]
    if not stats_row.empty:
        stats_list = stats_row.iloc[0][["num_nodes", "num_train", "num_relations", "num_valid", "num_test"]].tolist()
        arg_dict = update_stats(stats_list, arg_dict, config_dict, opt="stats_dataset")
    else:
        raise RuntimeError("Unrecognized dataset")

    return arg_dict


def update_stats(stats, arg_dict, config_dict, opt="stats"):
    keys_common = ["general.num_nodes", "general.num_train"]
    for i in range(len(keys_common)):
        k = keys_common[i]
        if arg_dict.get(k) is None and config_dict.get(k) != stats[i]:
            arg_dict.update({k: stats[i]})

    if opt == "stats_dataset":
        keys = ["general.num_relations", "general.num_valid", "general.num_test"]
        for i in range(len(keys)):
            k = keys[i]
            if arg_dict.get(k) is None and config_dict.get(k) != stats[i + 2]:
                arg_dict.update({k: stats[i + 2]})
    else:
        if arg_dict.get("general.num_edges") is None and config_dict.get("general.num_edges") != stats[2]:
            arg_dict.update({"general.num_edges": str(int(stats[2]))})

    return arg_dict


def update_data_path(dir, arg_dict):
    dir = Path(dir)

    if arg_dict.get("path.train_edges") is None:
        arg_dict.update({"path.train_edges": str(dir / Path("train_edges.pt"))})

    if arg_dict.get("custom_ordering"):
        arg_dict.update({"path.custom_ordering": str(dir / Path("custom_ordering.txt"))})

    if arg_dict.get("partitions_train"):
        arg_dict.update({"path.train_edges_paritions": str(dir / Path("train_edges_partitions.txt"))})

    if arg_dict.get("partitions_valid") and arg_dict.get("general.num_valid") != "0":
        arg_dict.update({"path.validation_edges_paritions": str(dir / Path("validation_edges_partitions.txt"))})

    if arg_dict.get("partitions_test") and arg_dict.get("general.num_test") != "0":
        arg_dict.update({"path.test_edges_paritions": str(dir / Path("test_edges_partitions.txt"))})

    if arg_dict.get("general.learning_task") is None:
        if arg_dict.get("general.num_valid") != "0" and arg_dict.get("path.validation_edges") is None:
            arg_dict.update({"path.validation_edges": str(dir / Path("valid_edges.pt"))})

        if arg_dict.get("general.num_test") != "0" and arg_dict.get("path.test_edges") is None:
            arg_dict.update({"path.test_edges": str(dir / Path("test_edges.pt"))})

        if arg_dict.get("path.node_ids") is None:
            arg_dict.update({"path.node_ids": str(dir / Path("node_mapping.txt"))})

        if arg_dict.get("general.num_relations") != "1" and arg_dict.get("path.relation_ids") is None:
            arg_dict.update({"path.relation_ids": str(dir / Path("rel_mapping.txt"))})
    else:
        if arg_dict.get("path.train_nodes") is None:
            arg_dict.update({"path.train_nodes": str(dir / Path("train_nodes.pt"))})

        if arg_dict.get("path.node_features") is None:
            arg_dict.update({"path.node_features": str(dir / Path("features.pt"))})

        if arg_dict.get("path.node_labels") is None:
            arg_dict.update({"path.node_labels": str(dir / Path("labels.pt"))})

        if arg_dict.get("general.num_valid") != "0" and arg_dict.get("path.valid_nodes") is None:
            arg_dict.update({"path.valid_nodes": str(dir / Path("valid_nodes.pt"))})

        if arg_dict.get("general.num_test") != "0" and arg_dict.get("path.test_nodes") is None:
            arg_dict.update({"path.test_nodes": str(dir / Path("test_nodes.pt"))})

    return arg_dict


def set_args():
    parser = argparse.ArgumentParser(
        description="Generate configs",
        prog="config_generator",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=("Specify certain config (optional): " + "[--<section>.<key>=<value>]"),
    )
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "output_directory",
        metavar="output_directory",
        type=str,
        help="Directory to put configs \nAlso "
        + "assumed to be the default directory of preprocessed"
        + " data if --data_directory is not specified",
    )
    parser.add_argument(
        "--data_directory", metavar="data_directory", type=str, help="Directory of the preprocessed data"
    )
    mode.add_argument("--dataset", "-d", metavar="dataset", type=str, help="Dataset to preprocess")
    mode.add_argument(
        "--stats",
        "-s",
        metavar=("num_nodes", "num_train"),
        nargs=2,
        help="Dataset statistics.\n"
        + "Enter in order of num_nodes, num_train.\n"
        + "This option will be overwritten if general.num_nodes "
        + "and/or general.num_train are/is specified.",
    )
    mode.add_argument(
        "--stats_nc",
        "-s_nc",
        metavar=("num_nodes", "num_train", "num_edges"),
        nargs=3,
        help="Enter in order of num_nodes, num_train, "
        + "num_edges if the learning task is node "
        + "classification.\n"
        + "This option will be overwritten if general.num_edges"
        + ", general.num_nodes, general.num_train is specified.",
    )
    parser.add_argument(
        "--device",
        "-dev",
        metavar="generate_config",
        choices=["GPU", "CPU", "multi-GPU"],
        nargs="?",
        default="GPU",
        help=(
            "Generates configs for a single-GPU/multi-CPU"
            + "/multi-GPU training configuration file by "
            + "default. \nValid options (default to GPU): "
            + "[GPU, CPU, multi-GPU]\n"
            + "This option will be overwritten if "
            + "general.device is specified."
        ),
    )
    parser.add_argument(
        "--custom_ordering",
        "-co",
        action="store_true",
        help="If stated, will add default custom_ordering "
        + "file path to configuration file.\n"
        + "This option will be overwritten if "
        + "path.custom_ordering is specified.",
    )
    parser.add_argument(
        "--partitions_train",
        action="store_true",
        help="If stated, will add default training edges "
        + "partitions file to configuration file.\n"
        + "This option will be overwritten if "
        + "path.train_edges_partitions is specified.",
    )
    parser.add_argument(
        "--partitions_valid",
        action="store_true",
        help="If stated, will add default valid edges "
        + "partitions file to configuration file.\n"
        + "This option will be overwritten if "
        + "path.validation_edges_partitions is specified.",
    )
    parser.add_argument(
        "--partitions_test",
        action="store_true",
        help="If stated, will add default test edges "
        + "partitions file to configuration file.\n"
        + "This option will be overwritten if "
        + "path.test_edges_partitions is specified.",
    )

    config_dict, valid_dict = read_template(DEFAULT_CONFIG_FILE)

    for key in list(config_dict.keys()):
        if valid_dict.get(key) is not None:
            parser.add_argument(
                str("--" + key),
                metavar=key,
                type=str,
                choices=valid_dict.get(key),
                default=config_dict.get(key),
                help=argparse.SUPPRESS,
            )
        else:
            parser.add_argument(
                str("--" + key), metavar=key, type=str, default=config_dict.get(key), help=argparse.SUPPRESS
            )

    return parser, config_dict


def parse_args(args, config_dict):
    arg_dict = vars(args)
    set_up_files(args.output_directory)

    for key in list(config_dict.keys()):
        if arg_dict.get(key) == config_dict.get(key):
            arg_dict.pop(key)

    if arg_dict.get("general.device") is None:
        if arg_dict.get("device") != config_dict.get("general.device"):
            arg_dict.update({"general.device": arg_dict.get("device")})

    if arg_dict.get("dataset") is not None:
        arg_dict = update_dataset_stats(arg_dict.get("dataset"), arg_dict, config_dict)
    elif arg_dict.get("stats") is not None:
        arg_dict = update_stats(arg_dict.get("stats"), arg_dict)
    elif arg_dict.get("stats_nc") is not None:
        arg_dict = update_stats(arg_dict.get("stats_nc"), arg_dict, config_dict, "nodeclassification")
    else:
        raise RuntimeError("Must specify either dataset or dataset stats.")

    dir = args.output_directory
    if args.data_directory is None:
        arg_dict = update_data_path(dir, arg_dict)
    else:
        arg_dict = update_data_path(args.data_directory, arg_dict)

    return arg_dict


def main():
    parser, config_dict = set_args()
    args = parser.parse_args()
    config_dict = parse_args(args, config_dict)
    output_config(config_dict, args.output_directory)


if __name__ == "__main__":
    main()
