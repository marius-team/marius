import os
import json
import argparse
import random
import time
import datetime

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
    parser.add_argument("--sample_percentage", default = 50, type = int, help = "The percentage of batches to sample to generate CDF")
    parser.add_argument("--log_rate", type=int, default=20, help="Log rate of the nodes processed")
    return parser.parse_args()

def main():
    start_time = time.time()
    arguments = read_arguments()
    config = read_config_file(arguments.config_file)

    # Create the loaders
    data_loader = DatasetLoader(config)
    features_loader = get_featurizer(data_loader, config["features_stats"])
    sampler = SubgraphSampler(data_loader, features_loader, config)

    # Determine the sample nodes
    sample_percentage = arguments.sample_percentage
    all_nodes = data_loader.get_nodes_sorted_by_incoming()
    batch_size = int(config["batch_size"])
    all_nodes = all_nodes.reshape((-1, batch_size))

    num_sample_nodes = int((sample_percentage * all_nodes.shape[0])/100.0) 
    batches_to_keep = np.random.choice(all_nodes.shape[0], num_sample_nodes, replace = False)
    sample_nodes = all_nodes[batches_to_keep, : ]
    print("Processing", sample_nodes.shape[0], "batches with each batch having", batch_size, "nodes")

    # Perform the sampling
    log_rate = int(sample_nodes.shape[0]/arguments.log_rate)
    pages_loaded = []
    for idx, curr_batch in enumerate(sample_nodes):
        average_pages_loaded = sampler.perform_sampling_for_nodes(curr_batch)
        if average_pages_loaded > 0:
            pages_loaded.append(average_pages_loaded)

        if idx > 0 and idx % log_rate == 0:
            percentage_finished = (100.0 * idx)/sample_nodes.shape[0]
            print("Finished processing", round(percentage_finished), "percent of nodes")

    # Get the arguments to log
    total_time = time.time() - start_time
    vals_to_log = {
        "Total Processing Time" : str(datetime.timedelta(seconds = int(total_time))),
        "Batch Size" : batch_size,
        "Sample Percentage" : sample_percentage,
    }

    for curr_obj in [data_loader, features_loader, sampler]:
        vals_to_log.update(curr_obj.get_values_to_log())

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

if __name__ == "__main__":
    main()