import os
import json
import argparse
import random
import time
import datetime
import traceback
import torch.multiprocessing as multiprocessing

from src.dataset_loader import *
from src.features_loader import *
from src.sampler import *
from src.visualizer import *

def read_config_file(config_file):
    with open(config_file, "r") as reader:
        return json.load(reader)

def run_for_worker(i, arguments, results_queue):
    # Create the loaders
    config = read_config_file(arguments.config_file)
    data_loader = DatasetLoader(config)
    features_loader = get_featurizer(data_loader, config["features_stats"])
    sampler = SubgraphSampler(data_loader, features_loader, config)

    # Determine the chunk for the current workers
    sample_percentage = 100
    if "sample_percentage" in config:
        sample_percentage = float(config["sample_percentage"])
    all_nodes = data_loader.get_nodes_sorted_by_incoming()
    total_nodes = all_nodes.shape[0]
    chunk_start_idx, chunk_end_idx = int((i * total_nodes)/arguments.num_process), int(((i + 1) * total_nodes)/arguments.num_process)
    worker_chunk = all_nodes[chunk_start_idx : chunk_end_idx]
    np.random.shuffle(worker_chunk)

    # Get the sample for this worker
    batch_size = 1
    if "batch_size" in config:
        batch_size = int(config["batch_size"])
    print("Worker", i, "is processing", worker_chunk.shape, "out of", all_nodes.shape, "using batch size of", batch_size)
    worker_chunk = worker_chunk[ : worker_chunk.shape[0] - (worker_chunk.shape[0] % batch_size)]
    worker_chunk = worker_chunk.reshape((-1, batch_size))
    num_sample_nodes = int((sample_percentage * worker_chunk.shape[0])/100.0) 
    batches_to_keep = np.random.choice(worker_chunk.shape[0], num_sample_nodes, replace = False)
    sample_nodes = worker_chunk[batches_to_keep, : ]

    # Perform the sampling
    total_batches = sample_nodes.shape[0]
    log_rate = int(total_batches/arguments.log_rate)
    print("Total of", total_batches, "batches with log rate of", log_rate)
    pages_loaded = []
    pages_time_taken = []
    for batch_idx, batch in enumerate(sample_nodes):
        # Process the current batch
        try:
            start_time = time.time()
            sucess, average_pages_loaded = sampler.perform_sampling_for_nodes(batch)
            if sucess and average_pages_loaded >= 0:
                pages_loaded.append(average_pages_loaded)
            pages_time_taken.append(time.time() - start_time)
        except:
            print("Worker", i, "batch", batch_idx, "failed due to error", traceback.format_exc())

        # Log the value
        if batch_idx > 0 and batch_idx % log_rate == 0:
            percentage_finished = (100.0 * batch_idx)/total_batches
            print("Worker", i, "finished processing", round(percentage_finished), "percent of nodes")

    # Get the arguments to log
    metrics = sampler.get_metrics()
    metrics["average_time_per_batch"] = np.mean(np.array(pages_time_taken))
    vals_to_log = {
        "Batch Size" : batch_size,
        "Sample Percentage" : sample_percentage,
    }
    for curr_obj in [data_loader, features_loader, sampler]:
        vals_to_log.update(curr_obj.get_values_to_log())
    
    results_queue.put({
        "pages_loaded" : np.array(pages_loaded),
        "vals_to_log" : vals_to_log,
        "sampling_depth" : config["sampling_depth"],
        "dataset_name" : config["dataset_name"],
        "metrics" : metrics
    })

def read_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_file", type=str, help="The config file containing the details for the simulation")
    parser.add_argument("--save_path", required=True, type=str, help="The path to save the resulting image to")
    parser.add_argument("--graph_title", required=True, type=str, help="The title of the saved graph")
    parser.add_argument("--log_rate", type=int, default = 20, help="Log rate of the nodes processed")
    parser.add_argument("--num_process", type=int, default = 1, help="Number of processes we want to use for processing")
    return parser.parse_args()

def main():
    start_time = time.time()
    arguments = read_arguments()
    multiprocessing.set_start_method('spawn')
    torch.set_num_threads(arguments.num_process)

    # Start the workers
    results_queue = multiprocessing.Queue()
    running_process = []
    for i in range(1, arguments.num_process):
        curr_process = multiprocessing.Process(target = run_for_worker, args = (i, arguments, results_queue, ))
        curr_process.start()
        running_process.append(curr_process)
    
    # Wait for all of the processes to finish
    run_for_worker(0, arguments, results_queue)
    [proc.join() for proc in running_process]
    total_time = int(time.time() - start_time)
    print("Got result in time:", str(datetime.timedelta(seconds = total_time)))

    # Read the result from each worker
    vals_to_log = {}
    metrics = {}
    dataset_name, sampling_depth = "", 0
    workers_results_gotten = 0
    all_workers_pages_loaded = []
    while workers_results_gotten < arguments.num_process:
        # Process the worker result
        process_result = results_queue.get()
        if workers_results_gotten == 0:
            vals_to_log.update(process_result["vals_to_log"])
            dataset_name, sampling_depth = process_result["dataset_name"], process_result["sampling_depth"]

        all_workers_pages_loaded.append(process_result["pages_loaded"])
        workers_results_gotten += 1

        # Record the metric
        for metric_name, metric_value in process_result["metrics"].items():
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append(metric_value)

    # Save the histogram
    all_pages_loaded = np.concatenate(all_workers_pages_loaded)
    os.makedirs(os.path.dirname(arguments.save_path), exist_ok=True)
    visualize_arguments = {
        "pages_loaded": all_pages_loaded,
        "save_path": arguments.save_path,
        "graph_title": arguments.graph_title,
        "depth" : sampling_depth,
        "dataset_name": dataset_name,
        "values_to_log": vals_to_log,
        "metrics" : metrics
    }
    visualize_results(visualize_arguments)

if __name__ == "__main__":
    main()