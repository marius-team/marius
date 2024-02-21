import os
import json
import argparse
import random
import time

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
    parser.add_argument("--num_nodes", default = -1, type = int, help = "The number of nodes we want in our sample")
    parser.add_argument("--log_rate", type=int, default=20, help="Log rate of the nodes processed")
    return parser.parse_args()

def main():
    start_time = time.time()
    arguments = read_arguments()
    config = read_config_file(arguments.config_file)

    # Create the loaders
    data_loader = DatasetLoader(config)
    print(data_loader.get_num_nodes(), data_loader.get_num_edges())
    '''
    features_loader = get_featurizer(data_loader, config["features_stats"])
    sampler = SubgraphSampler(data_loader, features_loader, config)
    print("Finished loading all objects")

    # Perform sampling
    nodes_to_sample = [i for i in range(data_loader.get_num_nodes())]
    random.shuffle(nodes_to_sample)
    if arguments.num_nodes > 0:
        nodes_to_sample = nodes_to_sample[ : arguments.num_nodes]
    log_rate = int(len(nodes_to_sample) / arguments.log_rate)

    pages_loaded = []
    for curr_node in nodes_to_sample:
        num_pages_read = sampler.perform_sampling_for_node(curr_node)
        if num_pages_read > 0:
            pages_loaded.append(num_pages_read)

        if len(pages_loaded) > 0 and len(pages_loaded) % log_rate == 0:
            percentage_finished = (100.0 * len(pages_loaded)) / len(nodes_to_sample)
            print("Finished processing", round(percentage_finished), "percent of nodes")

    # Get the arguments to log
    vals_to_log = dict()
    for curr_obj in [data_loader, features_loader, sampler]:
        vals_to_log.update(curr_obj.get_values_to_log())
    
    # Log the time taken
    total_time = time.time() - start_time
    print("Processed all", len(nodes_to_sample), "nodes in", total_time, "seconds")

    # Save the histogram
    os.makedirs(os.path.dirname(arguments.save_path), exist_ok=True)
    visualize_arguments = {
        "pages_loaded": pages_loaded,
        "save_path": arguments.save_path,
        "graph_title": arguments.graph_title,
        "depth" : config["sampling_depth"],
        "dataset_name": config["dataset_name"],
        "values_to_log": vals_to_log,
    }
    visualize_results(visualize_arguments)
    '''

if __name__ == "__main__":
    main()