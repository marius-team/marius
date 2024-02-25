import os
import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt

configs_dir = "configs/arvix_in_mem_benchmark"
save_dir = "results/arvix_in_mem_benchmark"
command_format = "python3 main.py --config_file {} --save_path {} --graph_title \"CDF for sequential features\""
def run_for_combo(sampling_depth, in_mem_percent):
    # Create the config file
    file_name = f"arvix_{sampling_depth}_hop_{in_mem_percent}_in_mem"
    config_file_path = os.path.join(configs_dir, file_name + ".json")
    if not os.path.exists(config_file_path):
        config_data = {
            "dataset_name" : "ogbn_arxiv",
            "features_stats" : {
                "featurizer_type" : "default",
                "page_size" : "16.384 KB",
                "feature_dimension" : 128,
                "feature_size" : "float32"
            }, 
            "batch_size" : 25,
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
    
    mean_val = float(metrics["Mean Pages Loaded"])
    std_dev = float(metrics["Std dev of Pages Loaded"])
    return [sampling_depth, in_mem_percent, mean_val, std_dev]

def main():
    all_possible_depths = [i for i in range(1, 4)]
    all_in_mem_percent = [i/2.0 for i in range(1, 11)] + [10, 25, 50, 75, 100]
    
    results_rows = []
    for sampling_depth in all_possible_depths:
        for in_memory_percent in all_in_mem_percent:
            results_rows.append(run_for_combo(sampling_depth, in_memory_percent))
    
    result_df = pd.DataFrame(results_rows, columns = ["sampling_depth", "percent_in_mem", "mean_pages", "std_dev_pages_loaded"])
    result_df.to_csv("arvix_sampling_in_mem_benchmark.csv", index = False)

def visualize_results():
    # Create the graph
    colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']
    color_idx = 0
    fig, ax = plt.subplots()

    df = pd.read_csv("arvix_sampling_in_mem_benchmark.csv")
    for sampling_depth, rows in df.groupby("sampling_depth"):
        # Get the values for this depth
        x_values = rows["percent_in_mem"].values
        mean_val = rows["mean_pages"].values
        std_dev = rows["std_dev_pages_loaded"].values
        print("X values of", x_values)

        color = colors[color_idx]
        color_idx += 1
        ax.plot(x_values, mean_val, '-', label = "Sampling depth " + str(sampling_depth), color = color)
        ax.fill_between(x_values, mean_val - std_dev, mean_val + std_dev, alpha=0.2, color = color)
    
    ax.set_ylim((0.5, 250))
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Percent of nodes in cache")
    ax.set_ylabel("Pages loaded")
    ax.legend()
    fig.tight_layout()
    plt.savefig("arvix_sampling_in_mem_benchmark.png")
    
if __name__ == "__main__":
    # main()
    visualize_results()