import json

import cycler
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mticker

import parse_output
from buffer_simulator import plotting as plot_buff
import pandas as pd


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return idx


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
    hits10 = complex_res["Hits@10"][-1]
    time = sum(complex_res["Train Time"]) / 1000.0

    print("Marius Complex: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))

    MRR = distmul_res["MRR"][-1]
    hits1 = distmul_res["Hits@1"][-1]
    hits10 = distmul_res["Hits@10"][-1]
    time = sum(distmul_res["Train Time"]) / 1000.0

    print("Marius DistMult: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))

    dglke_complex = exp_dir + "dgl-ke/complex_fb15k_result.json"
    dglke_distmult = exp_dir + "dgl-ke/distmult_fb15k_result.json"

    with open(dglke_complex) as f:
        complex_res = json.load(f)
    with open(dglke_distmult) as f:
        distmul_res = json.load(f)

    MRR = complex_res["MRR"]
    hits1 = complex_res["Hits@1"]
    hits10 = complex_res["Hits@10"]
    time = complex_res["Train Time"]

    print("DGL-KE Complex: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))

    MRR = distmul_res["MRR"]
    hits1 = distmul_res["Hits@1"]
    hits10 = distmul_res["Hits@10"]
    time = distmul_res["Train Time"]

    print("DGL-KE DistMult: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))

    pbg_complex = exp_dir + "pbg/complex_fb15k_result.json"
    pbg_distmult = exp_dir + "pbg/distmult_fb15k_result.json"

    with open(pbg_complex) as f:
        complex_res = json.load(f)
    with open(pbg_distmult) as f:
        distmul_res = json.load(f)

    MRR = complex_res["MRR"]
    hits1 = complex_res["Hits@1"]
    hits10 = complex_res["Hits@10"]
    time = complex_res["Train Time"]

    print("PBG Complex: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))

    MRR = distmul_res["MRR"]
    hits1 = distmul_res["Hits@1"]
    hits10 = distmul_res["Hits@10"]
    time = distmul_res["Train Time"]

    print("PBG DistMult: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))


def print_table_3():
    exp_dir = "osdi2021/system_comparisons/livejournal/"

    marius_dot = exp_dir + "marius/dot_live_journal_result.json"

    with open(marius_dot) as f:
        marius_dot_res = json.load(f)

    MRR = marius_dot_res["MRR"][-1]
    hits1 = marius_dot_res["Hits@1"][-1]
    hits5 = marius_dot_res["Hits@5"][-1]
    hits10 = marius_dot_res["Hits@10"][-1]
    time = sum(marius_dot_res["Train Time"]) / 1000.0

    print("Marius Dot: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))

    pbg_dot = exp_dir + "pbg/dot_live_journal_result.json"

    with open(pbg_dot) as f:
        pbg_dot_res = json.load(f)

    MRR = pbg_dot_res["MRR"][-1]
    hits1 = pbg_dot_res["Hits@1"][-1]
    hits5 = marius_dot_res["Hits@5"][-1]
    hits10 = pbg_dot_res["Hits@10"][-1]
    time = pbg_dot_res["Train Time"]

    print("PBG Dot: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))

    dglke_dot = exp_dir + "dgl-ke/dot_live_journal_result.json"

    with open(dglke_dot) as f:
        dglke_dot_res = json.load(f)

    MRR = dglke_dot_res["MRR"]
    hits1 = dglke_dot_res["Hits@1"]
    hits10 = dglke_dot_res["Hits@10"]
    time = dglke_dot_res["Train Time"]

    print("DGL-KE Dot: MRR %s, Hits@1 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits10, time))


def print_table_4():
    exp_dir = "osdi2021/system_comparisons/twitter/"

    marius_dot = exp_dir + "marius/dot_twitter_result.json"

    with open(marius_dot) as f:
        marius_dot_res = json.load(f)

    MRR = marius_dot_res["MRR"][-1]
    hits1 = marius_dot_res["Hits@1"][-1]
    hits5 = marius_dot_res["Hits@5"][-1]
    hits10 = marius_dot_res["Hits@10"][-1]
    time = sum(marius_dot_res["Train Time"]) / 1000.0

    print("Marius Dot: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))

    pbg_dot = exp_dir + "pbg/dot_twitter_result.json"

    with open(pbg_dot) as f:
        pbg_dot_res = json.load(f)

    MRR = pbg_dot_res["MRR"][-1]
    hits1 = pbg_dot_res["Hits@1"][-1]
    hits5 = marius_dot_res["Hits@5"][-1]
    hits10 = pbg_dot_res["Hits@10"][-1]
    time = pbg_dot_res["Train Time"]

    print("PBG Dot: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))


def print_table_5():
    exp_dir = "osdi2021/system_comparisons/freebase86m/"

    marius_complex = exp_dir + "marius/freebase86m_16_result.json"

    with open(marius_complex) as f:
        marius_complex_res = json.load(f)

    MRR = marius_complex_res["MRR"][-1]
    hits1 = marius_complex_res["Hits@1"][-1]
    hits5 = marius_complex_res["Hits@5"][-1]
    hits10 = marius_complex_res["Hits@10"][-1]
    time = sum(marius_complex_res["Train Time"]) / 1000.0

    print("Marius Complex, P=16: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (
    MRR, hits1, hits5, hits10, time))

    pbg_complex = exp_dir + "pbg/freebase86m_16_result.json"

    with open(pbg_complex) as f:
        pbg_complex_res = json.load(f)

    MRR = pbg_complex_res["MRR"][-1]
    hits1 = pbg_complex_res["Hits@1"][-1]
    hits5 = pbg_complex_res["Hits@5"][-1]
    hits10 = pbg_complex_res["Hits@10"][-1]
    time = pbg_complex_res["Train Time"]

    print("PBG ComplEx: MRR %s, Hits@1 %s, Hits@5 %s, Hits@10 %s, Runtime %s s" % (MRR, hits1, hits5, hits10, time))


def print_table_6():
    exp_dir = "osdi2021/large_embeddings/"

    d20 = exp_dir + "d20_result.json"
    d50 = exp_dir + "d50_result.json"
    d100 = exp_dir + "d100_result.json"

    with open(d20) as f:
        d20_res = json.load(f)

    with open(d50) as f:
        d50_res = json.load(f)

    with open(d100) as f:
        d100_res = json.load(f)

    MRR = d20_res["MRR"][-1]
    time = d20_res["Train Time"][0]

    print("Marius D=20, P=16: MRR %s, Runtime %s s" % (MRR, time))

    MRR = d50_res["MRR"][-1]
    time = d50_res["Train Time"][0]

    print("Marius D=50, P=16: MRR %s, Runtime %s s" % (MRR, time))

    MRR = d100_res["MRR"][-1]
    time = d100_res["Train Time"][0]

    print("Marius D=100, P=16: MRR %s, Runtime %s s" % (MRR, time))

def plot_figure_7():
    exp_dir = "osdi2021/buffer_simulator/"

    n_start = 8
    c_start = 2
    num = 5
    total_size = 86E6 * 4 * 2 * 100 # total embedding size for freebase86m d=100
    plot_buff.plot_varying_num_partitions_io(n_start, c_start, num, total_size, exp_dir + "figure7.png")
    print("Figure written to %s" % exp_dir + "figure7.png")


def plot_figure_8():
    exp_dir = "osdi2021/system_comparisons/freebase86m/"

    marius8_df = pd.read_csv(exp_dir + "marius/complex_50_8_util_nvidia_smi.csv")
    marius_mem_df = pd.read_csv(exp_dir + "marius/complex_50_util_nvidia_smi.csv")
    # marius8_dstat_df = pd.read_csv(exp_dir + "marius/complex_50_8_util_dstat.csv")
    # marius_mem_dstat_df = pd.read_csv(exp_dir + "marius/complex_50_util_dstat.csv")

    # dglke_df = parse_output.read_nvidia_smi(exp_dir + "dgl-ke/complex_50_util_nvidia_smi.csv")
    # dglke_dstat_df = parse_output.read_dstat(exp_dir + "dgl-ke/complex_50_util_dstat.csv")

    pbg_df = pd.read_csv(exp_dir + "pbg/complex_50_8_util_nvidia_smi.csv")
    # pbg_dstat_df = pd.read_csv(exp_dir + "pbg/complex_50_8_util_dstat.csv")

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
    # gpus_dfs.append(dglke_df)

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

    # orders = [2, 3, 1, 0]
    orders = [1, 2, 0]

    # experiment_ids = ["Marius (On Disk, 8 Partitions)", "Marius (In Memory)", "PBG (On Disk, 8 Partitions)",
    #                   "DGL-KE (In Memory)"]
    experiment_ids = ["Marius (On Disk, 8 Partitions)", "Marius (In Memory)", "PBG (On Disk, 8 Partitions)"]
    # experiment_ids = ["PBG", "DGL-KE"]
    experiment_handles = []
    i = 0
    for gpu_res in gpus_dfs:
        # cpu_res = dstat_dfs[i]
        gpu_util = smooth(gpu_res["GPU Compute Utilization"].to_numpy(), 30) * 100.0
        # cpu_util = smooth(cpu_res["CPU User Utilization"].to_numpy() + cpu_res["CPU Sys Utilization"].to_numpy(), 25)

        color = color_cycler.by_key()['color'][i]

        offset = offsets[i]
        ax.plot(np.arange(len(gpu_util[offset[0]:offset[1]])), gpu_util[offset[0]:offset[1]], color=color, linewidth=4,
                linestyle="-", label=experiment_ids[i], zorder=orders[i])
        ax.set_ylabel("GPU Util (%)")
        ax.set_xlabel("Time (s)")

        i += 1

    f.set_tight_layout(True)

    plt.legend()
    plt.ylim(0, 100)

    plt.savefig(exp_dir + "figure_8.png")

    print("Plot saved to: %s" % exp_dir + "figure_8.png")


def plot_figure_9():

    exp_dir = "osdi2021/partition_orderings/freebase86m/"

    elimination = exp_dir + "elimination100_util_dstat.csv"
    hilbert = exp_dir + "hilbert100_util_dstat.csv"
    hilbert_sym = exp_dir + "hilbertsymmetric100_util_dstat.csv"

    elimination_dstat = pd.read_csv(elimination)
    hilbert_dstat = pd.read_csv(hilbert)
    hilbert_sym_dstat = pd.read_csv(hilbert_sym)

    elimination_total_io = elimination_dstat["Bytes Read"] + elimination_dstat["Bytes Written"]
    hilbert_total_io = hilbert_dstat["Bytes Read"] + hilbert_dstat["Bytes Written"]
    hilbert_sym_total_io = hilbert_sym_dstat["Bytes Read"] + hilbert_sym_dstat["Bytes Written"]

    elimination_total_io = np.cumsum(elimination_total_io / np.power(2, 30))
    hilbert_total_io = np.cumsum(hilbert_total_io / np.power(2, 30))
    hilbert_sym_total_io = np.cumsum(hilbert_sym_total_io / np.power(2, 30))

    plt.plot(elimination_total_io, linestyle="--", linewidth=2, color="r", label="Elimination")
    plt.plot(hilbert_total_io, linestyle="-", linewidth=2, color="b", label="Hilbert")
    plt.plot(hilbert_sym_total_io, linestyle="-.", linewidth=2, color="g", label="Hilbert Symmetric")

    plt.ylabel("Total IO (GB)")
    plt.xlabel("Time (s)")

    plt.savefig(exp_dir + "figure_9.png")
    print("Plot saved to: %s" % exp_dir + "figure_9.png")


    # def find_nearest(a, a0):
    #     "Element in nd array `a` closest to the scalar value `a0`"
    #     idx = np.abs(a - a0).argmin()
    #     return idx
    #
    # plt.figure(num=None, figsize=(6, 2), facecolor='w', edgecolor='k')
    #
    # params = {'legend.fontsize': 9,
    #           'legend.handlelength': 4}
    # plt.rc('font', family='serif')
    # plt.rcParams['font.size'] = 9
    # plt.rcParams['axes.labelsize'] = 9
    # plt.rcParams['axes.labelweight'] = 'bold'
    # COLOR = 'black'
    # plt.rcParams['text.color'] = COLOR
    # plt.rcParams['axes.labelcolor'] = COLOR
    # plt.rcParams['xtick.color'] = COLOR
    # plt.rcParams['ytick.color'] = COLOR
    # plt.rcParams.update(params)
    #
    # exp_dir = "osdi2021/partition_orderings/freebase86m/"
    #
    # orderings100_results = {}
    #
    # orderings100_results["Hilbert"] = parse_output.read_nvidia_smi(exp_dir + "hilbert100_util_dstat.csv")
    # orderings100_results["HilbertSymmetric"] = parse_output.read_nvidia_smi(exp_dir + "hilbertsymmetric100_util_dstat.csv")
    # orderings100_results["Elimination"] = parse_output.read_nvidia_smi(exp_dir + "elimination100_util_dstat.csv")
    #
    # colors = ["r", "b", "g", "c"]
    #
    # zorders=[0, 2, 1]
    #
    # runtimes = [2039, 809, 1159]
    # # runtimes = [2060, 900, 1300]
    # start_times = [615, 560, 615]
    # offsets = []
    #
    # for i in range(3):
    #     offsets.append((start_times[i], start_times[i] + runtimes[i]))
    #
    # labels = ["Hilbert", "Elimination", "HilbertSymmetric"]
    # linestyles = ["--", "-", "-."]
    #
    # i = 0
    # for k, v in orderings100_results.items():
    #     y = np.cumsum(v[0].disk_util[offsets[i][0]:offsets[i][1]]) / np.power(2, 30)
    #     cpu_ts = v[0].cpu_trace_timestamp[offsets[i][0]:offsets[i][1]]
    #     iter_ts = np.asarray([np.datetime64(itr[0]) for itr in v[0].iteration_progress])
    #
    #     nearest = [find_nearest(iter_ts, c) for c in cpu_ts]
    #     # y = smooth(v[0].disk_util[offsets[i][0]:offsets[i][1]], 25) / np.power(2,30)
    #
    #     ls = linestyles[i]
    #
    #     print(k, y[-1])
    #     plt.plot(nearest, y, linestyle=ls, linewidth=2, color=colors[i], zorder=zorders[i], label=labels[i])
    #     # plt.plot(x[-1], y[-1], markersize=12, marker="o", color = colors[i], zorder=zorders[i])
    #     i+=1
    #
    # plt.ylabel("Total IO (GB)")
    # # plt.ylim(0, 1500)
    # plt.xlabel("Iteration")
    # plt.legend(loc="upper left")
    # plt.gcf().subplots_adjust(bottom=0.3)
    # plt.show()
    # # plt.savefig("../../../../figures/orderings_freebase86m.pdf")


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

    mem50 = exp_dir + "memory50_result.json"

    elim50 = exp_dir + "elimination50_result.json"
    elim100 = exp_dir + "elimination100_result.json"

    hilbert50 = exp_dir + "hilbert50_result.json"
    hilbert100 = exp_dir + "hilbert100_result.json"

    hilbertsymmetric50 = exp_dir + "hilbertsymmetric50_result.json"
    hilbertsymmetric100 = exp_dir + "hilbertsymmetric100_result.json"

    # mem = [77, 0]
    # elim = [100, 150]
    # hilbert_sym = [130, 207]
    # hilbert = [250, 354]

    mem = []
    elim = []
    hilbert_sym = []
    hilbert = []

    with open(mem50) as f:
        mem.append(sum(json.load(f)["Train Time"]) / 1000.0)
        mem.append(0)

    with open(elim50) as f:
        elim.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(elim100) as f:
        elim.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbert50) as f:
        hilbert.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbert100) as f:
        hilbert.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbertsymmetric50) as f:
        hilbert_sym.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbertsymmetric100) as f:
        hilbert_sym.append(sum(json.load(f)["Train Time"]) / 1000.0)


    # d100 = [0, 150, 207, 354]
    colors = ["k", "b", "g", "r"]

    ind = np.arange(len(mem)) / 2.25  # the x locations for the groups
    width = 0.1  # the width of the bars

    rects1 = ax.bar(ind - (width / 2) - width, mem, width, label='In Memory', color=color_cycler.by_key()['color'][0])
    rects2 = ax.bar(ind - (width / 2), elim, width, label='Elimination', color=color_cycler.by_key()['color'][3])

    rects1 = ax.bar(ind + (width / 2), hilbert_sym, width, label='HilbertSymmetric',
                    color=color_cycler.by_key()['color'][6])
    rects2 = ax.bar(ind + (width / 2) + width, hilbert, width, label='Hilbert',
                    color=color_cycler.by_key()['color'][10])

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    ax.set_ylabel('Runtime (s)', fontsize=16, fontdict={'weight': "bold", "family": 'serif'})
    ax.set_xticks(ind)
    ax.set_xticklabels(('d=50', 'd=100'), fontsize=16, fontdict={'weight': "bold", "family": 'serif'})
    ax.legend(loc="upper left")
    fig.tight_layout()
    plt.savefig(exp_dir + "figure_10.png")
    print("Plot saved to: %s" % exp_dir + "figure_10.png")


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


    # elim = [222, 322]
    # hilbert_sym = [223, 410]
    # hilbert = [236, 495]
    # d100 = [0, 150, 207, 354]

    elim50 = exp_dir + "elimination100_result.json"
    elim100 = exp_dir + "elimination200_result.json"

    hilbert50 = exp_dir + "hilbert100_result.json"
    hilbert100 = exp_dir + "hilbert200_result.json"

    hilbertsymmetric50 = exp_dir + "hilbertsymmetric100_result.json"
    hilbertsymmetric100 = exp_dir + "hilbertsymmetric200_result.json"


    elim = []
    hilbert_sym = []
    hilbert = []

    with open(elim50) as f:
        elim.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(elim100) as f:
        elim.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbert50) as f:
        hilbert.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbert100) as f:
        hilbert.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbertsymmetric50) as f:
        hilbert_sym.append(sum(json.load(f)["Train Time"]) / 1000.0)

    with open(hilbertsymmetric100) as f:
        hilbert_sym.append(sum(json.load(f)["Train Time"]) / 1000.0)

    ind = np.arange(len(elim)) / 2.25  # the x locations for the groups
    width = 0.1  # the width of the bars

    rects2 = ax.bar(ind - width, elim, width, label='Elimination', color=color_cycler.by_key()['color'][3])
    rects1 = ax.bar(ind, hilbert_sym, width, label='HilbertSymmetric', color=color_cycler.by_key()['color'][6])
    rects2 = ax.bar(ind + width, hilbert, width, label='Hilbert', color=color_cycler.by_key()['color'][10])

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Runtime (s)', fontsize=16)
    ax.set_xticks(ind)
    ax.set_xticklabels(("d=100", "d=200"), fontsize=16, fontdict={'weight': "bold", "family": 'serif'})
    ax.legend(loc="upper left")

    fig.tight_layout()

    plt.savefig(exp_dir + "figure_11.png")

    print("Plot saved to: %s" % exp_dir + "figure_11.png")


def plot_figure_12(bounds):
    fig, ax_arr = plt.subplots((2), figsize=(10, 4))
    ax1 = ax_arr[0]
    ax2 = ax_arr[1]
    # ax3 =ax_arr[2]

    plt.rc('font', family='serif', weight="bold")
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.labelweight'] = "normal"
    COLOR = 'black'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR

    xs = []
    ys = []

    exp_dir = "osdi2021/microbenchmarks/bounded_staleness/"

    with open(exp_dir + "all_sync_result.json") as f:
        all_sync_result = json.load(f)

    sync_rels_results = [all_sync_result]
    async_rels_results = [all_sync_result]

    all_bounds = [0] + bounds

    for bound in all_bounds[1:]:
        with open(exp_dir + "sync_rel_%i_result.json" % bound) as f:
            sync_rels_results.append(json.load(f))
        with open(exp_dir + "all_async_%i_result.json" % bound) as f:
            async_rels_results.append(json.load(f))

    for i, v in enumerate(sync_rels_results):
        x = all_bounds[i]
        y = v["MRR"][-1]
        xs.append(x)
        ys.append(y)

    l = sorted((i, j) for i, j in zip(xs, ys))
    new_x, new_y = zip(*l)
    ax1.plot(new_x, new_y, "r-", marker="o", lw=4, markersize=12)

    xs = []
    ys = []
    for i, v in enumerate(async_rels_results):
        x = all_bounds[i]
        y = v["MRR"][-1]
        xs.append(x)
        ys.append(y)

    l = sorted((i, j) for i, j in zip(xs, ys))
    new_x, new_y = zip(*l)
    ax1.plot(new_x, new_y, "b-", marker="x", lw=4, markersize=12)

    y = all_sync_result["MRR"][-1]
    ax1.axhline(y, color="k", linestyle="--", lw=4)

    ax1.set_xscale("log", basex=2)
    ax1.set_ylim([.2, .8])

    labels = [item.get_text() for item in ax1.get_xticklabels()]

    empty_string_labels = [''] * len(labels)
    ax1.set_xticklabels(empty_string_labels)

    xs = []
    ys = []
    for i, v in enumerate(sync_rels_results):
        x = all_bounds[i]
        y = v["TP"][-1]
        xs.append(x)
        ys.append(y)

    l = sorted((i, j) for i, j in zip(xs, ys))
    new_x, new_y = zip(*l)
    ax2.plot(new_x, new_y, "r", marker="o", linestyle="--", lw=4, markersize=12)

    xs = []
    ys = []
    for i, v in enumerate(async_rels_results):
        x = all_bounds[i]
        y = v["TP"][-1]
        xs.append(x)
        ys.append(y)

    l = sorted((i, j) for i, j in zip(xs, ys))
    new_x, new_y = zip(*l)
    ax2.plot(new_x, new_y, "b", marker="x", linestyle="--", lw=4, markersize=12)

    y = all_sync_result["TP"][-1]
    ax2.axhline(y, color="k", linestyle="--", lw=4)

    ax2.set_ylim([10000, 1000000])

    ax2.set_xscale("log", basex=2)
    ax2.set_xlabel('Staleness Bound', fontsize=20, fontdict={'weight' : "bold", "family": 'serif'})
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, y: '{}'.format(int(x))))
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))
    fig.tight_layout()

    custom_lines = [Line2D([0], [0], marker="o", color='r', lw=4, markersize=12),
                    Line2D([0], [0], marker="x", color="b", lw=4, markersize=12),
                    Line2D([0], [0], linestyle="--", color="k", lw=4, markersize=12)]

    ax1.legend(custom_lines, ['Sync Relations', 'Async Relations', "All Sync"], loc="lower left")

    ax1.tick_params(axis="x", labelsize=14)
    ax1.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="x", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)

    ax1.set_ylabel('MRR', fontsize=20, fontdict={"family": 'serif', "weight": "bold"})
    ax2.set_ylabel('Edges/sec', fontsize=20, fontdict={"family": 'serif', "weight": "bold"})

    ax2.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.set_yticklabels(ax2.get_yticks(), fontdict={"family": 'serif', "weight": "normal"})
    plt.rc('font', family='serif', weight="normal")
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.10e' % x))
    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(g))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)

    plt.savefig(exp_dir + "figure_12.png")

    print("Plot saved to: %s" % exp_dir + "figure_12.png")


def print_figure_13():
    exp_dir = "osdi2021/microbenchmarks/prefetching/"

    prefetching_result = exp_dir + "prefetching_result.json"
    no_prefetching_result = exp_dir + "no_prefetching_result.json"

    with open(prefetching_result) as f:
        prefetching_res = json.load(f)
    with open(no_prefetching_result) as f:
        no_prefetching_res = json.load(f)

    time = sum(prefetching_res["Train Time"]) / 1000.0
    print("Prefetching: Runtime %s s" % (time))

    time = sum(no_prefetching_res["Train Time"]) / 1000.0
    print("No Prefetching: Runtime %s s" % (time))


