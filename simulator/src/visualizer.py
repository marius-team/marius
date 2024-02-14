import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_results(visualize_args, num_bins=60, x_range=(0, 75), write_location=(0.55, 0.6)):
    # Get the number of pages read
    pages_loaded = visualize_args["pages_loaded"]
    np_arr = np.array(pages_loaded)
    page_mean = round(np.mean(np_arr), 2)
    page_std = round(np.std(np_arr), 2)

    # Create the histogram
    plt.figure()
    plt.ecdf(pages_loaded, label="CDF")
    plt.hist(pages_loaded, bins=num_bins, histtype="step", density=True, cumulative=True, label="Cumulative histogram")
    plt.xlabel("Number of pages loaded for node inference")
    plt.ylabel("Percentage of nodes")
    plt.title(visualize_args["graph_title"] + " for dataset " + visualize_args["dataset_name"])
    plt.xlim(x_range)
    plt.legend()

    # Write some resulting text
    text_to_write = "Mean Pages Loaded: " + str(page_mean) + "\n"
    text_to_write += "Std dev of Pages Loaded: " + str(page_std) + "\n"
    text_to_write += "Feature File Size: " + visualize_args["total_space"] + "\n"
    text_to_write += "Node Features per Page: " + str(visualize_args["nodes_per_page"])

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
    plt.tight_layout()
    plt.savefig(visualize_args["save_path"])
