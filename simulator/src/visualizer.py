import matplotlib.pyplot as plt
import os


def visualize_results(pages_loaded, save_path, dataset_name, num_bins=50):
    # Create the histogram
    plt.figure()
    plt.ecdf(pages_loaded, label="CDF")
    plt.hist(pages_loaded, bins=num_bins, histtype="step", density=True, cumulative=True, label="Cumulative histogram")
    plt.xlabel("Number of pages loaded for node inference")
    plt.ylabel("Percentage of nodes")
    plt.title("Number of pages loaded for node inference on " + dataset_name)
    plt.xlim(0, 50)
    plt.legend()

    # Save the result
    print("Saving the result to", save_path)
    plt.savefig(save_path)
