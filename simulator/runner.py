import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import concurrent.futures

configs_dir = "configs/ogbn_products_benchmark"
save_dir = "results/ogbn_products_benchmark"
command_format = "python3 main.py --config_file {} --save_path {} --graph_title \"CDF for sequential features\""
def run_for_combo(sampling_depth, in_mem_percent):
    # Create the config file
    file_name = f"products_{sampling_depth}_hop_{in_mem_percent}_in_mem"
    config_file_path = os.path.join(configs_dir, file_name + ".json")
    if not os.path.exists(config_file_path):
        config_data = {
            "dataset_name" : "ogbn_products",
            "features_stats" : {
                "featurizer_type" : "linear",
                "page_size" : "16.384 KB",
                "feature_dimension" : 100,
                "feature_size" : "float32"
            }, 
            "batch_size" : 5,
            "sample_percentage" : 20,
            "sampling_depth" : sampling_depth,
            "top_percent_in_mem" : in_mem_percent
        }

        with open(config_file_path, 'w+') as writer:
            json.dump(config_data, writer)
    
    # Run the file
    img_save_path = os.path.join(save_dir, file_name + ".png")
    metrics_path = os.path.join(save_dir, file_name + ".json")
    if not os.path.exists(metrics_path):
        command = command_format.format(config_file_path, img_save_path)
        print("Running command", command)
        subprocess.run(command, shell = True, capture_output = True, text = True)
    
    # Read the metrics
    with open(metrics_path, 'r') as reader:
        metrics = json.load(reader)
    
    mean_val = float(metrics["pages_loaded_mean"])
    std_dev = float(metrics["pages_loaded_std_dev"])
    return [sampling_depth, in_mem_percent, mean_val, std_dev]

def main(num_workers = 5):
    # Create the all in memory percent config
    os.makedirs(configs_dir, exist_ok = True)
    os.makedirs(save_dir, exist_ok = True)

    all_possible_depths = [i for i in range(1, 4)]
    all_in_mem_percent = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    
    results_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = num_workers) as executor:
        future_to_result = {}
        # Submmit the futures
        for sampling_depth in all_possible_depths:
            for in_memory_percent in all_in_mem_percent:
                future = executor.submit(run_for_combo, sampling_depth, in_memory_percent)
                future_to_result[future] = (sampling_depth, in_memory_percent)

        # Get the results
        for future in concurrent.futures.as_completed(future_to_result):
            try:
                results_rows.append(future.result())
            except Exception as exc:
                print('%r generated an exception: %s' % (future_to_result[future], exc))

    result_df = pd.DataFrame(results_rows, columns = ["sampling_depth", "percent_in_mem", "mean_pages", "std_dev_pages_loaded"])
    result_df = result_df.sort_values(by = ["sampling_depth", "percent_in_mem"])
    result_df.to_csv("ogbn_products_in_mem_benchmark.csv", index = False)

def visualize_results():
    # Create the graph
    colors = ['tab:blue', 'tab:green', 'tab:orange']
    color_idx = 0
    fig, ax = plt.subplots()

    df = pd.read_csv("ogbn_products_in_mem_benchmark.csv")
    for sampling_depth, rows in df.groupby("sampling_depth"):
        # Get the values for this depth
        rows = rows.sort_values(by = ["percent_in_mem"])
        x_values = rows["percent_in_mem"].values
        mean_val = rows["mean_pages"].values
        std_dev = rows["std_dev_pages_loaded"].values

        color = colors[color_idx]
        color_idx += 1
        ax.plot(x_values, mean_val, '-', label = "Sampling depth " + str(sampling_depth), color = color)
        ax.fill_between(x_values, mean_val - std_dev, mean_val + std_dev, alpha=0.2, color = color)
    
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Percent of nodes in cache")
    ax.set_ylabel("Avg Pages loaded for inference")
    ax.set_title("Impact of in memory caching on ogbn_products dataset")
    ax.legend()
    fig.tight_layout()
    plt.savefig("ogbn_products_in_mem_benchmark.png")
    
if __name__ == "__main__":
    main()
    visualize_results()