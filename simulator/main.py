import os
import json
import argparse
import random

from src.dataset_loader import *
from src.features_loader import *
from src.sampler import *
from src.visualizer import *


def read_config_file(config_file):
    with open(config_file, "r") as reader:
        return json.load(reader)


def read_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, help="The config file containing the details for the simulation")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the resulting image to")
    parser.add_argument("--graph_title", required=True, type=str, help="The title of the saved graph")
    return parser.parse_args()


def main():
    arguments = read_arguments()
    config = read_config_file(arguments.config_file)

    # Create the loaders
    data_loader = DatasetLoader(config["dataset_name"])
    features_loader = FeaturesLoader(data_loader, config["features_stats"])
    sampler = SubgraphSampler(data_loader, features_loader)

    # Perform sampling
    nodes_to_sample = [i for i in range(data_loader.get_num_nodes())]
    random.shuffle(nodes_to_sample)

    pages_loaded = []
    for curr_node in nodes_to_sample:
        num_pages_read = sampler.perform_sampling_for_node(curr_node)
        if num_pages_read > 0:
            pages_loaded.append(num_pages_read)
    print("Got result for", len(pages_loaded), "nodes out of", len(nodes_to_sample), "nodes")

    # Save the histogram
    os.makedirs(os.path.dirname(arguments.save_path), exist_ok=True)
    visualize_results(pages_loaded, arguments.save_path, arguments.graph_title, config["dataset_name"])


if __name__ == "__main__":
    main()
