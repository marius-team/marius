from pathlib import Path
import argparse
import pandas as pd
import os

HERE = os.path.abspath(os.path.dirname(__file__))
DEFAULT_CONFIG_FILE = os.path.join(HERE, "config_templates",
                                   "default_configs.txt")
DATASET_STATS = os.path.join(HERE, "dataset_stats", "dataset_stats.tsv")


def output_config(config_dict, output_dir):
    device = config_dict.get("device")
    ds_name = config_dict.get("dataset")

    file = Path(output_dir) / Path(str(ds_name) + "_" +
                                   device.lower() + ".ini")
    sections = ["model", "storage", "training", "training_pipeline",
                "evaluation", "evaluation_pipeline", "path", "reporting"]
    opts = list(config_dict.keys())

    with open(file, "w+") as f:
        f.write("[general]\n")
        f.write("device=" + config_dict.get("general.device") + "\n")
        f.write("gpu_ids=" + config_dict.get("general.gpu_ids") + "\n")
        if config_dict.get("general.random_seed") is not None:
            f.write("random_seed=" + config_dict.get("general.random_seed")
                    + "\n")
        f.write("num_train=" + str(config_dict.get("num_train")) + "\n")
        f.write("num_nodes=" + str(config_dict.get("num_nodes")) + "\n")
        f.write("num_relations=" + str(config_dict.get("num_relations"))
                + "\n")
        f.write("num_valid=" + str(config_dict.get("num_valid")) + "\n")
        f.write("num_test=" + str(config_dict.get("num_test")) + "\n")
        f.write("experiment_name=" +
                config_dict.get("general.experiment_name") + "\n")

        for sec in sections:
            f.write("\n[" + sec + "]\n")
            for key in opts:
                if key.split(".")[0] == sec:
                    f.write(key.split(".")[1] +
                            "=" + config_dict.get(key) + "\n")


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
            valid_dict.update({line[0]: [sub_line[1:]]})
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


def update_dataset_stats(dataset, config_dict):
    datasets_stats = pd.read_csv(DATASET_STATS, sep='\t')
    stats_row = datasets_stats[datasets_stats['dataset'] == dataset]
    if not stats_row.empty:
        stats_list = stats_row.iloc[0][['num_nodes', 'num_relations',
                                        'num_train', 'num_valid',
                                        'num_test']].tolist()
        config_dict = update_stats(stats_list, config_dict)
    else:
        raise RuntimeError("Unrecognized dataset")

    return config_dict


def update_stats(stats, config_dict):
    config_dict.update({"num_train": str(int(stats[2]))})
    config_dict.update({"num_nodes": str(int(stats[0]))})
    config_dict.update({"num_relations": str(int(stats[1]))})
    config_dict.update({"num_valid": str(int(stats[3]))})
    config_dict.update({"num_test": str(int(stats[4]))})

    return config_dict


def update_data_path(dir, config_dict):
    config_dict.update({"path.train_edges": str(dir.strip("/") +
                        "/train_edges.pt")})
    config_dict.update({"path.train_edges_partitions": str(dir.strip("/") +
                        "/train_edges_partitions.txt")})
    config_dict.update({"path.valid_edges": str(dir.strip("/") +
                        "/valid_edges.pt")})
    config_dict.update({"path.test_edges": str(dir.strip("/") +
                        "/test_edges.pt")})
    config_dict.update({"path.node_labels": str(dir.strip("/") +
                        "/node_mapping.txt")})
    config_dict.update({"path.relation_labels": str(dir.strip("/") +
                        "/rel_mapping.txt")})
    config_dict.update({"path.node_ids": str(dir.strip("/") +
                        "/node_mapping.bin")})
    config_dict.update({"path.relation_ids": str(dir.strip("/") +
                        "/rel_mapping.bin")})

    return config_dict


def set_args():
    parser = argparse.ArgumentParser(
                description='Generate configs', prog='config_generator',
                formatter_class=argparse.RawTextHelpFormatter,
                epilog=('Specify certain config (optional): ' +
                        '[--<section>.<key>=<value>]'))
    mode = parser.add_mutually_exclusive_group()
    parser.add_argument('output_directory', metavar='output_directory',
                        type=str, help='Directory to put configs \nAlso ' +
                        'assumed to be the default directory of preprocessed' +
                        ' data if --data_directory is not specified')
    parser.add_argument('--data_directory', metavar='data_directory',
                        type=str, help='Directory of the preprocessed data')
    mode.add_argument('--dataset', '-d', metavar='dataset', type=str,
                      help='Dataset to preprocess')
    mode.add_argument('--stats', '-s',
                      metavar=('num_nodes', 'num_relations', 'num_train',
                               'num_valid', 'num_test'),
                      nargs=5, help='Dataset statistics\n' +
                      'Enter in order of num_nodes, num_relations, num_train' +
                      ' num_valid, num_test')
    parser.add_argument('--device', '-dev', metavar='generate_config',
                        choices=["GPU", "CPU", "multi-GPU"],
                        nargs='?', default='GPU',
                        help=('Generates configs for a single-GPU/multi-CPU' +
                              '/multi-GPU training configuration file by ' +
                              'default. \nValid options (default to GPU): ' +
                              '[GPU, CPU, multi-GPU]'))

    config_dict, valid_dict = read_template(DEFAULT_CONFIG_FILE)

    for key in list(config_dict.keys())[1:]:
        if valid_dict.get(key) is not None:
            parser.add_argument(str("--" + key), metavar=key, type=str,
                                choices=valid_dict.get(key),
                                default=config_dict.get(key),
                                help=argparse.SUPPRESS)
        else:
            parser.add_argument(str("--" + key), metavar=key, type=str,
                                default=config_dict.get(key),
                                help=argparse.SUPPRESS)

    return parser, config_dict


def parse_args(args):
    arg_dict = vars(args)
    set_up_files(args.output_directory)

    arg_dict.update({"general.device": arg_dict.get("device")})
    if arg_dict.get("device") == "multi-GPU":
        arg_dict.update({"device": "multi_GPU"})
    else:
        arg_dict.update({"device": arg_dict.get("device")})

    if arg_dict.get("general.random_seed") == "#":
        arg_dict.pop("general.random_seed")

    if arg_dict.get("dataset") is not None:
        arg_dict.update({"dataset": arg_dict.get("dataset")})
        arg_dict = update_dataset_stats(arg_dict.get("dataset"), arg_dict)
    elif arg_dict.get("stats") is not None:
        arg_dict = update_stats(arg_dict.get("stats"), arg_dict)
    else:
        raise RuntimeError("Must specify either dataset or dataset stats.")

    return arg_dict


if __name__ == "__main__":
    parser, config_dict = set_args()
    args = parser.parse_args()
    config_dict = parse_args(args)

    dir = args.output_directory
    if args.data_directory is None:
        config_dict = update_data_path(dir, config_dict)
    else:
        config_dict = update_data_path(args.data_directory, config_dict)

    output_config(config_dict, dir)
