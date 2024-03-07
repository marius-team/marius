import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
import numpy as np
import json
import torch

def visualize_results(visualize_args, num_bins=60, write_location=(0.65, 0.5)):
    # Get the number of pages read
    pages_loaded = visualize_args["pages_loaded"]
    page_mean = round(np.mean(pages_loaded), 2)
    page_std = round(np.std(pages_loaded), 2)
    print("Got pages loaded of mean", page_mean, "and std_dev of", page_std)

    # Plot the ecdf
    ecdf = sm.distributions.ECDF(pages_loaded)    
    num_samples = int((np.max(pages_loaded) - np.min(pages_loaded))/1.25)
    ecdf_x = np.linspace(np.min(pages_loaded), np.max(pages_loaded), num = num_samples)
    ecdf_y = ecdf(ecdf_x)
    plt.step(ecdf_x, ecdf_y, label="ECDF")

    # Plot the hisotgram
    plt.hist(pages_loaded, bins=num_bins, histtype="step", density=True, cumulative=True, label="Cumulative histogram")
    plt.xlabel("Number of pages loaded for node inference")
    plt.ylabel("Percentage of nodes")
    title = visualize_args["graph_title"] + " for dataset " + visualize_args["dataset_name"] + " (Sampling depth = " + str(visualize_args["depth"]) + ")"
    plt.title(title, fontsize = 10)
    plt.legend()

    # Write some resulting text
    vals_to_log = {
        "Mean Pages Loaded" : page_mean,
        "Std dev of Pages Loaded" : page_std
    }
    if "values_to_log" in visualize_args:
        vals_to_log.update(visualize_args["values_to_log"])
    
    metrics_to_write = {
        "pages_loaded_mean" : page_mean,
        "pages_loaded_std_dev" : page_std
    }
    if "metrics" in visualize_args:
        for metric_name, metric_value in visualize_args["metrics"].items():
            metrics_to_write[metric_name] = metric_value

    txt_lines = []
    for key, value in vals_to_log.items():
        txt_lines.append(str(key).strip() + ": " + str(value).strip())
    text_to_write = "Key Metrics:\n" + "\n".join(txt_lines)

    # Get the current axis limits
    xlim = plt.xlim()
    ylim = plt.ylim()
    actual_x = write_location[0] * (xlim[1] - xlim[0]) + xlim[0]
    actual_y = write_location[1] * (ylim[1] - ylim[0]) + ylim[0]
    plt.text(
        actual_x,
        actual_y,
        text_to_write,
        fontsize=10,
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(facecolor="red", alpha=0.5),
    )

    # Save the result
    image_save_path = visualize_args["save_path"]
    plt.savefig(image_save_path)
    print("Saving image to", image_save_path)

    # Save the metrics
    metrics_save_path = image_save_path[ : image_save_path.rindex(".")] + ".json"
    print("Saving metrics to", metrics_save_path)
    with open(metrics_save_path, 'w+') as writer:
        json.dump(metrics_to_write, writer, indent = 4)
    
    # Save the tensor
    tensor_save_path = image_save_path[ : image_save_path.rindex(".")] + ".pt"
    print("Saving tensor to", tensor_save_path)
    torch.save(visualize_args["all_batches"], tensor_save_path)