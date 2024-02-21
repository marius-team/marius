import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_results(visualize_args, num_bins=60, write_location=(0.7, 0.6)):
    # Get the number of pages read
    pages_loaded = visualize_args["pages_loaded"]
    np_arr = np.array(pages_loaded)
    page_mean = round(np.mean(np_arr), 2)
    page_std = round(np.std(np_arr), 2)
    pages_upper_bound = int(np.percentile(pages_loaded, 99))

    # Create the histogram
    plt.figure()
    plt.ecdf(pages_loaded, label="CDF")
    plt.hist(pages_loaded, bins=num_bins, histtype="step", density=True, cumulative=True, label="Cumulative histogram")
    plt.xlabel("Number of pages loaded for node inference")
    plt.ylabel("Percentage of nodes")
    title = visualize_args["graph_title"] + " for dataset " + visualize_args["dataset_name"] + " (Sampling depth = " + str(visualize_args["depth"]) + ")"
    plt.title(title, fontsize = 10)
    plt.xlim((0, pages_upper_bound))
    plt.legend()

    # Write some resulting text
    txt_lines = ["Mean Pages Loaded: " + str(page_mean), "Std dev of Pages Loaded: " + str(page_std)]
    if "values_to_log" in visualize_args:
        vals_to_log = visualize_args["values_to_log"]
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
    plt.tight_layout()
    plt.savefig(visualize_args["save_path"])
