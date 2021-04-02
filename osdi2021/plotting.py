import matplotlib.pyplot as plt
import numpy as np
import parse_output
import cycler
from matplotlib.lines import Line2D
import json

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def print_table_2():
    exp_dir = "osdi2021/system_comparisons/fb15k/"

    marius_complex = exp_dir + "marius/complex_fb15k_result.json"
    marius_distmult = exp_dir + "marius/distmult_fb15k_result.json"

    with open(marius_complex) as f:
        complex_res = json.load(f)
    with open(marius_distmult) as f:
        distmul_res = json.load(f)

    MRR = complex_res["MRR"][-1]
    hits1 = complex_res["Hits@1"][-1]
    hits5 = complex_res["Hits@5"][-1]
    hits10 = complex_res["Hits@10"][-1]
    time = sum(complex_res["Train Time"]) / 1000.0

    print("Marius Complex: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))

    MRR = distmul_res["MRR"][-1]
    hits1 = distmul_res["Hits@1"][-1]
    hits5 = distmul_res["Hits@5"][-1]
    hits10 = distmul_res["Hits@10"][-1]
    time = sum(distmul_res["Train Time"]) / 1000.0

    print("Marius DistMult: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))

    pass

def print_table_3():
    pass

def print_table_4():
    pass

def print_table_5():
    pass

def print_table_6():
    pass

def plot_figure_7():
    pass

def plot_figure_8():
    exp_dir = "osdi2021/system_comparisons/freebase86m/"

    marius8_df = parse_output.read_nvidia_smi(exp_dir + "marius/complex_50_8_util_nvidia_smi.csv")
    marius_mem_df = parse_output.read_nvidia_smi(exp_dir + "marius/complex_50_util_nvidia_smi.csv")
    marius8_dstat_df = parse_output.read_dstat(exp_dir + "marius/complex_50_8_util_dstat.csv")
    marius_mem_dstat_df = parse_output.read_dstat(exp_dir + "marius/complex_50_util_dstat.csv")

    dglke_df = parse_output.read_nvidia_smi(exp_dir + "dgl-ke/complex_50_util_nvidia_smi.csv")
    dglke_dstat_df = parse_output.read_dstat(exp_dir + "dgl-ke/complex_50_util_dstat.csv")

    pbg_df = parse_output.read_nvidia_smi(exp_dir + "pbg/complex_50_8_util_nvidia_smi.csv")
    pbg_dstat_df = parse_output.read_dstat(exp_dir + "pbg/complex_50_8_util_dstat.csv")

    params = {'legend.fontsize': 12,
              'legend.handlelength': 4}
    plt.rc('font', family='serif')
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR
    plt.rcParams.update(params)


    gpus_dfs = []
    gpus_dfs.append(marius8_df)
    gpus_dfs.append(marius_mem_df)
    gpus_dfs.append(pbg_df)
    gpus_dfs.append(dglke_df)

    dstat_dfs = []
    dstat_dfs.append(marius8_dstat_df)
    dstat_dfs.append(marius_mem_dstat_df)
    dstat_dfs.append(pbg_dstat_df)


    offsets = []
    # to get proper offsets for each line these need to be tuned
    # offsets.append((375, 675))
    # offsets.append((400, 650))
    # offsets.append((0, 990))
    # offsets.append((1800, 3200))
    offsets.append((0, -1))
    offsets.append((0, -1))
    offsets.append((0, -1))
    offsets.append((0, -1))


    color = plt.cm.viridis(np.linspace(0, 1, 4))
    color_cycler = cycler.cycler('color', color)

    f, ax = plt.subplots(figsize=(8, 4))

    i = 0

    orders = [2, 3, 1, 0]

    experiment_ids = ["Marius (On Disk, 8 Partitions)", "Marius (In Memory)", "PBG (On Disk, 8 Partitions)", "DGL-KE (In Memory)"]
    # experiment_ids = ["PBG", "DGL-KE"]
    experiment_handles = []
    for e in experiment_ids:


        gpu_res = gpus_dfs[i]
        # cpu_res = dstat_dfs[i]
        gpu_util = smooth(gpu_res["GPU Compute Utilization"].to_numpy(), 30) * 100.0
        # cpu_util = smooth(cpu_res["CPU User Utilization"].to_numpy() + cpu_res["CPU Sys Utilization"].to_numpy(), 25)

        color = color_cycler.by_key()['color'][i]

        offset = offsets[i]
        ax.plot(np.arange(len(gpu_util[offset[0]:offset[1]])), gpu_util[offset[0]:offset[1]], color=color, linewidth=4, linestyle = "-", label=e, zorder=orders[i])
        ax.set_ylabel("GPU Util (%)")
        ax.set_xlabel("Time (s)")


        i += 1

    f.set_tight_layout(True)

    experiment_handles.append(Line2D([0], [0], color=color, label=e))

    plt.legend()
    plt.ylim(0, 100)

    plt.savefig(exp_dir + "figure_8.png")

def plot_figure_9():
    pass

def plot_figure_10():
    exp_dir = "osdi2021/partition_orderings/freebase86m/"

    labels = []
    fig, ax = plt.subplots(figsize=(6, 4))


    plt.rc('font', family='serif')
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    colors = plt.cm.viridis(np.linspace(0, 1, 16))
    color_cycler = cycler.cycler('color', colors)


    mem = [77, 0]
    elim = [100, 150]
    hilbert_sym = [130, 207]
    hilbert = [250, 354]
    # d100 = [0, 150, 207, 354]
    colors = ["k", "b", "g", "r"]

    ind = np.arange(len(mem)) / 2.25 # the x locations for the groups
    width = 0.1  # the width of the bars


    rects1 = ax.bar(ind - (width / 2) - width, mem, width, label='In Memory', color=color_cycler.by_key()['color'][0])
    rects2 = ax.bar(ind - (width / 2), elim, width, label='Elimination', color=color_cycler.by_key()['color'][3])

    rects1 = ax.bar(ind + (width / 2), hilbert_sym, width, label='HilbertSymmetric', color=color_cycler.by_key()['color'][6])
    rects2 = ax.bar(ind + (width / 2) + width, hilbert, width, label='Hilbert', color=color_cycler.by_key()['color'][10])


    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    ax.set_ylabel('Runtime (m)', fontsize=16, fontdict={'weight' : "bold", "family": 'serif'})
    ax.set_xticks(ind)
    ax.set_xticklabels(('d=50', 'd=100'), fontsize=16, fontdict={'weight' : "bold", "family": 'serif'})
    ax.set_ylim([0, 500])
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(exp_dir + "figure_10.png")
    # plt.show()

def plot_figure_11():
    exp_dir = "osdi2021/partition_orderings/twitter/"

    fig, ax = plt.subplots(figsize=(6, 4))
    plt.rc('font', family='serif')
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    colors = plt.cm.viridis(np.linspace(0, 1, 16))
    color_cycler = cycler.cycler('color', colors)

    elim = [222, 322]
    hilbert_sym = [223, 410]
    hilbert = [236, 495]
    # d100 = [0, 150, 207, 354]
    colors = ["k", "b", "g", "r"]

    ind = np.arange(len(elim)) / 2.25  # the x locations for the groups
    width = 0.1  # the width of the bars


    rects2 = ax.bar(ind - width, elim, width, label='Elimination', color=color_cycler.by_key()['color'][3])
    rects1 = ax.bar(ind, hilbert_sym, width, label='HilbertSymmetric', color=color_cycler.by_key()['color'][6])
    rects2 = ax.bar(ind + width, hilbert, width, label='Hilbert', color=color_cycler.by_key()['color'][10])

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Runtime (m)', fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(("d=100", "d=200"), fontsize=16, fontdict={'weight' : "bold", "family": 'serif'})
    ax.legend(loc="upper left")
    ax.set_ylim([0, 550])

    fig.tight_layout()

    plt.savefig(exp_dir + "figure_11.png")

def plot_figure_12():
    pass

def plot_figure_13():
    pass