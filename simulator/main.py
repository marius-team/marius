from src.dataset_loader import *
from src.sampler import *
from src.visualizer import *

import os
import json
import argparse
import random
import time
import datetime
import traceback
import torch
import torch.multiprocessing as multiprocessing

def read_config_file(config_file):
    with open(config_file, "r") as reader:
        return json.load(reader)

def run_for_worker(arguments):
    # Create the loaders
    config = read_config_file(arguments.config_file)
    data_loader = DatasetLoader(config)
    sampler = SubgraphSampler(data_loader, config)

    # Read the config arguments
    sample_percentage = 100
    if "sample_percentage" in config:
        sample_percentage = float(config["sample_percentage"])

    batch_size = 1
    if "batch_size" in config:
        batch_size = int(config["batch_size"])

    # Get the sample batches
    worker_chunk = data_loader.get_nodes_sorted_by_incoming() 
    num_sample_nodes = int((sample_percentage * worker_chunk.size(0))/100.0) 
    node_probabilities = torch.full((worker_chunk.size(0), ), 1.0, dtype = torch.float)
    batches_idx = torch.multinomial(node_probabilities, num_sample_nodes, replacement = False)
    worker_chunk = worker_chunk[batches_idx]
    worker_chunk = worker_chunk[ : worker_chunk.size(0) - (worker_chunk.size(0) % batch_size)]
    worker_chunk = worker_chunk[torch.randperm(worker_chunk.size(0))]
    sample_nodes = worker_chunk.reshape((-1, batch_size))

    # Perform the sampling
    total_batches = sample_nodes.shape[0]
    log_rate = int(total_batches/arguments.log_rate)
    print("Total of", total_batches, "batches with log rate of", log_rate)
    pages_loaded = []
    pages_time_taken = []
    batch_results = []
    for batch_idx, batch in enumerate(sample_nodes):
        # Process the current batch
        try:
            start_time = time.time()
            sucess, average_pages_loaded = sampler.perform_sampling_for_nodes(batch)
            if sucess and average_pages_loaded >= 0:
                avg_tensor = torch.tensor([average_pages_loaded])
                batch_results.append(torch.cat((avg_tensor, batch), 0))
                pages_loaded.append(average_pages_loaded)

            pages_time_taken.append(time.time() - start_time)
        except:
            print("Worker", i, "batch", batch_idx, "failed due to error", traceback.format_exc())

        # Log the value
        if batch_idx > 0 and batch_idx % log_rate == 0:
            percentage_finished = (100.0 * batch_idx)/total_batches
            print("Finished processing", round(percentage_finished), "percent of nodes")

    # Create all batches    
    all_batches = torch.stack(batch_results)
    sort_idx = all_batches[ : , 0].argsort(dim = 0, descending = True)
    all_batches = all_batches[sort_idx]

    # Create the metrics
    metrics = sampler.get_metrics()
    node_avg_time = np.mean(np.array(pages_time_taken))/batch_size
    metrics["average_time_per_node"] = node_avg_time
    metrics["estimated_total_time_minutes"] = (node_avg_time * data_loader.get_num_nodes())/60.0
    vals_to_log = {
        "Batch Size" : batch_size,
        "Sample Percentage" : sample_percentage,
    }
    for curr_obj in [data_loader, sampler]:
        vals_to_log.update(curr_obj.get_values_to_log())
    
    # Save the result and write to disk
    resulting_values = {
        "pages_loaded" : pages_loaded,
        "vals_to_log" : vals_to_log,
        "sampling_depth" : config["sampling_depth"],
        "dataset_name" : config["dataset_name"],
        "all_batches" : all_batches,
        "metrics" : metrics
    }

    return resulting_values

def read_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, help="The config file containing the details for the simulation")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the resulting image to")
    parser.add_argument("--graph_title", required=True, type=str, help="The title of the saved graph")
    parser.add_argument("--log_rate", type=int, default = 20, help="Log rate of the nodes processed")
    return parser.parse_args()

def main():
    start_time = time.time()
    arguments = read_arguments()
    os.makedirs(os.path.dirname(arguments.save_path), exist_ok = True)

    # Wait for all of the processes to finish
    process_result = run_for_worker(arguments)
    total_time = int(time.time() - start_time)
    time_taken = str(datetime.timedelta(seconds = total_time))
    print("Time taken of", time_taken)

    # Read the result from each worker
    vals_to_log = process_result["vals_to_log"]
    metrics = process_result["metrics"]
    dataset_name, sampling_depth = process_result["dataset_name"], process_result["sampling_depth"]
    pages_loaded = np.array(process_result["pages_loaded"])
    all_batches = process_result["all_batches"]

    # Save the histogram
    os.makedirs(os.path.dirname(arguments.save_path), exist_ok=True)
    visualize_arguments = {
        "pages_loaded": pages_loaded,
        "save_path": arguments.save_path,
        "graph_title": arguments.graph_title,
        "depth" : sampling_depth,
        "dataset_name": dataset_name,
        "values_to_log": vals_to_log,
        "metrics" : metrics,
        "all_batches" : all_batches
    }

    visualize_results(visualize_arguments)

if __name__ == "__main__":
    main()