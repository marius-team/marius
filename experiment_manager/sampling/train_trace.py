from marius.tools.preprocess.datasets.ogbn_papers100m import OGBNPapers100M
import marius as m
import marius.tools.configuration.marius_config as mc
from pathlib import Path
import torch
from omegaconf import OmegaConf
from tracing import NvidiaSmiMonitor
from parsing import parse_nvidia_smi
from executor import run_config
import multiprocessing as mp
import os

import time
import numpy as np
import pandas as pd
import shutil

CONFIG_FILE = "config_{}.yaml"
RESULT_FILE = "result_{}.csv"
OUTPUT_FILE = "out_{}.txt"

BASE_PATH = Path("experiment_manager/sampling/configs/")


def get_mean_max_gpu_mem(output_dir: Path):
    assert (output_dir / Path("nvidia_smi_{}.csv".format(0))).exists()
    nvidia_smi_df = parse_nvidia_smi(output_dir / Path("nvidia_smi_{}.csv".format(0)))
    return nvidia_smi_df["GPU Memory Used"].mean(), nvidia_smi_df["GPU Memory Used"].max()


def pyg_train_trace(config_path: Path,
                    output_dir: Path,
                    overwrite=True):

    if not overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Experiment already run. Results in {}".format(output_dir / Path(RESULT_FILE.format(0))))
        print("=========================================")
        return
    elif overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Overwriting previous experiment.")

    run_config(config_path, output_dir, overwrite, enable_nvidia_smi=True, system="PYG", output_to_terminal=False)

    # sampling_times = []
    loading_times = []
    transfer_times = []
    compute_times = []
    num_nodes = []
    num_edges = []

    with open(output_dir / Path(OUTPUT_FILE.format(0))) as f:
        lines = f.readlines()

        for line in lines:
            if "LOAD" in line:
                loading_times.append(float(line.split()[1]))
                transfer_times.append(float(line.split()[3]))
                compute_times.append(float(line.split()[5]))
                num_nodes.append(float(line.split()[7]))
                num_edges.append(float(line.split()[9]))

    # sampling_times = np.asarray(sampling_times)
    loading_times = np.asarray(loading_times)
    transfer_times = np.asarray(transfer_times)
    compute_times = np.asarray(compute_times)
    num_nodes = np.asarray(num_nodes)
    num_edges = np.asarray(num_edges)

    # mean_sampling = np.mean(sampling_times)
    mean_loading = np.mean(loading_times)
    mean_transfer = np.mean(transfer_times)
    mean_compute = np.mean(compute_times)
    mean_num_nodes = np.mean(num_nodes)
    mean_num_edges = np.mean(num_edges)

    mean_gpu, max_gpu = get_mean_max_gpu_mem(output_dir)

    dict_results = {
        "mean_sampling": [0],
        "mean_loading": [mean_loading],
        "mean_transfer": [mean_transfer],
        "mean_compute": [mean_compute],
        "num_nodes": [mean_num_nodes],
        "num_edges": [mean_num_edges],
        "mean_gpu_mem": [mean_gpu],
        "max_gpu_mem": [max_gpu]
    }

    results_df = pd.DataFrame(dict_results)
    results_df.to_csv(output_dir / Path(RESULT_FILE.format(0)), index=False)


def dgl_train_trace(config_path: Path,
                    output_dir: Path,
                    overwrite=True):

    if not overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Experiment already run. Results in {}".format(output_dir / Path(RESULT_FILE.format(0))))
        print("=========================================")
        return
    elif overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Overwriting previous experiment.")

    run_config(config_path, output_dir, overwrite, enable_nvidia_smi=True, system="DGL", output_to_terminal=False)

    # sampling_times = []
    loading_times = []
    transfer_times = []
    compute_times = []
    num_nodes = []
    num_edges = []

    with open(output_dir / Path(OUTPUT_FILE.format(0))) as f:
        lines = f.readlines()

        for line in lines:
            if "LOAD" in line:
                loading_times.append(float(line.split()[1]))
                transfer_times.append(float(line.split()[3]))
                compute_times.append(float(line.split()[5]))
                num_nodes.append(float(line.split()[7]))
                num_edges.append(float(line.split()[9]))

    # sampling_times = np.asarray(sampling_times)
    loading_times = np.asarray(loading_times)
    transfer_times = np.asarray(transfer_times)
    compute_times = np.asarray(compute_times)

    # mean_sampling = np.mean(sampling_times)
    mean_loading = np.mean(loading_times)
    mean_transfer = np.mean(transfer_times)
    mean_compute = np.mean(compute_times)
    mean_num_nodes = np.mean(num_nodes)
    mean_num_edges = np.mean(num_edges)

    mean_gpu, max_gpu = get_mean_max_gpu_mem(output_dir)

    dict_results = {
        "mean_sampling": [0],
        "mean_loading": [mean_loading],
        "mean_transfer": [mean_transfer],
        "mean_compute": [mean_compute],
        "num_nodes": [mean_num_nodes],
        "num_edges": [mean_num_edges],
        "mean_gpu_mem": [mean_gpu],
        "max_gpu_mem": [max_gpu]
    }

    results_df = pd.DataFrame(dict_results)
    results_df.to_csv(output_dir / Path(RESULT_FILE.format(0)), index=False)


def _marius_train_trace(config_path: Path,
                        output_dir: Path,
                        overwrite=True,
                        no_compute=False):

    model, dataloader = m.initialize_from_file(filename=config_path.__str__(),
                                               train=True,
                                               load_storage=True)

    with open(output_dir / Path(OUTPUT_FILE.format(0)), "w") as f:
        while dataloader.hasNextBatch():
            batch = dataloader.getNextBatch()

            t0 = time.time()
            dataloader.nodeClassificationSample(batch, 0)
            t1 = time.time()

            dataloader.loadCPUParameters(batch)
            t2 = time.time()

            if not no_compute:

                batch.to(model.current_device)
                torch.cuda.synchronize()
                t3 = time.time()

                model.train_batch(batch)
                t4 = time.time()

                num_nodes = batch.gnn_graph.node_ids.size(0)
                num_edges = batch.gnn_graph.src_sorted_edges.size(0) + batch.gnn_graph.dst_sorted_edges.size(0)

                f.write("SAMPLE {:.4f} LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}\n".format(t1 - t0,
                                                                                                              t2 - t1,
                                                                                                              t3 - t2,
                                                                                                              t4 - t3,
                                                                                                              num_nodes,
                                                                                                              num_edges
                                                                                                              ))
            else:
                batch.gnn_graph.performMap()
                num_nodes = batch.gnn_graph.node_ids.size(0)
                num_edges = batch.gnn_graph.src_sorted_edges.size(0) + batch.gnn_graph.dst_sorted_edges.size(0)

                batch.clear()

                f.write("SAMPLE {:.4f} LOAD {:.4f} TRANSFER {:.4f} COMPUTE {:.4f} NODES {} EDGES {}\n".format(t1 - t0,
                                                                                                              t2 - t1,
                                                                                                              -1,
                                                                                                              -1,
                                                                                                              num_nodes,
                                                                                                              num_edges
                                                                                                              ))


def marius_train_trace(config_path: Path,
                       output_dir: Path,
                       overwrite=True,
                       no_compute=False):
    print("=========================================")
    print("Running: {} \nConfiguration: {}\nSaving results to: {}".format("Marius", config_path, output_dir))

    os.makedirs(output_dir, exist_ok=True)

    if not overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Experiment already run. Results in {}".format(output_dir / Path(RESULT_FILE.format(0))))
        print("=========================================")
        return
    elif overwrite and (output_dir / Path(RESULT_FILE.format(0))).exists():
        print("Overwriting previous experiment.")

    t0 = time.time()
    nvidia_smi_monitor = NvidiaSmiMonitor(output_dir, 0)

    nvidia_smi_monitor.start()
    p = mp.Process(target=_marius_train_trace, args=(config_path, output_dir, overwrite, no_compute))
    p.start()
    p.join()
    nvidia_smi_monitor.stop()

    sampling_times = []
    loading_times = []
    transfer_times = []
    compute_times = []
    num_nodes = []
    num_edges = []

    with open(output_dir / Path(OUTPUT_FILE.format(0))) as f:
        lines = f.readlines()

        for line in lines:
            if "LOAD" in line:
                sampling_times.append(float(line.split()[1]))
                loading_times.append(float(line.split()[3]))
                transfer_times.append(float(line.split()[5]))
                compute_times.append(float(line.split()[7]))
                num_nodes.append(float(line.split()[9]))
                num_edges.append(float(line.split()[11]))

    sampling_times = np.asarray(sampling_times)
    loading_times = np.asarray(loading_times)
    transfer_times = np.asarray(transfer_times)
    compute_times = np.asarray(compute_times)
    num_nodes = np.asarray(num_nodes)
    num_edges = np.asarray(num_edges)

    mean_sampling = np.mean(sampling_times)
    mean_loading = np.mean(loading_times)
    mean_transfer = np.mean(transfer_times)
    mean_compute = np.mean(compute_times)
    mean_num_nodes = np.mean(num_nodes)
    mean_num_edges = np.mean(num_edges)

    mean_gpu, max_gpu = get_mean_max_gpu_mem(output_dir)

    dict_results = {
        "mean_sampling": [mean_sampling],
        "mean_loading": [mean_loading],
        "mean_transfer": [mean_transfer],
        "mean_compute": [mean_compute],
        "num_nodes": [mean_num_nodes],
        "num_edges": [mean_num_edges],
        "mean_gpu_mem": [mean_gpu],
        "max_gpu_mem": [max_gpu]
    }

    results_df = pd.DataFrame(dict_results)
    results_df.to_csv(output_dir / Path(RESULT_FILE.format(0)), index=False)

    shutil.copy(config_path, output_dir / Path(CONFIG_FILE.format(0)))

    t1 = time.time()
    print("Complete. Total runtime: {:.4f}s".format(t1 - t0))
    print("=========================================")


def run_train_trace(dataset_dir: Path, results_dir: Path, overwrite, enable_dstat, enable_nvidia_smi, show_output,
                    short, num_runs=1):
    dataset_name = "ogbn_papers100m"

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNPapers100M(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    mp.set_start_method("spawn")

    marius_1_layer_config = BASE_PATH / Path("marius_1_layer.yaml")
    marius_2_layer_config = BASE_PATH / Path("marius_2_layer.yaml")
    marius_3_layer_config = BASE_PATH / Path("marius_3_layer.yaml")
    marius_4_layer_config = BASE_PATH / Path("marius_4_layer.yaml")
    marius_5_layer_config = BASE_PATH / Path("marius_5_layer.yaml")

    marius_train_trace(marius_1_layer_config, results_dir / Path("sampling/marius_1_layer"), overwrite)
    marius_train_trace(marius_2_layer_config, results_dir / Path("sampling/marius_2_layer"), overwrite)
    marius_train_trace(marius_3_layer_config, results_dir / Path("sampling/marius_3_layer"), overwrite)
    marius_train_trace(marius_4_layer_config, results_dir / Path("sampling/marius_4_layer"), overwrite)
    marius_train_trace(marius_5_layer_config, results_dir / Path("sampling/marius_5_layer"), overwrite)
    marius_train_trace(marius_5_layer_config, results_dir / Path("sampling/marius_5_layer_no_compute"), overwrite, no_compute=True)

    dgl_1_layer_config = BASE_PATH / Path("dgl_1_layer.txt")
    dgl_2_layer_config = BASE_PATH / Path("dgl_2_layer.txt")
    dgl_3_layer_config = BASE_PATH / Path("dgl_3_layer.txt")
    dgl_4_layer_config = BASE_PATH / Path("dgl_4_layer.txt")
    dgl_5_layer_config = BASE_PATH / Path("dgl_5_layer.txt")
    dgl_5_layer_config_nc = BASE_PATH / Path("dgl_5_layer_no_compute.txt")

    dgl_train_trace(dgl_1_layer_config, results_dir / Path("sampling/dgl_1_layer"), overwrite)
    dgl_train_trace(dgl_2_layer_config, results_dir / Path("sampling/dgl_2_layer"), overwrite)
    dgl_train_trace(dgl_3_layer_config, results_dir / Path("sampling/dgl_3_layer"), overwrite)
    dgl_train_trace(dgl_4_layer_config, results_dir / Path("sampling/dgl_4_layer"), overwrite)
    dgl_train_trace(dgl_5_layer_config, results_dir / Path("sampling/dgl_5_layer"), overwrite)
    dgl_train_trace(dgl_5_layer_config_nc, results_dir / Path("sampling/dgl_5_layer_no_compute"), overwrite)

    pyg_1_layer_config = BASE_PATH / Path("pyg_1_layer.txt")
    pyg_2_layer_config = BASE_PATH / Path("pyg_2_layer.txt")
    pyg_3_layer_config = BASE_PATH / Path("pyg_3_layer.txt")
    pyg_4_layer_config = BASE_PATH / Path("pyg_4_layer.txt")
    pyg_5_layer_config = BASE_PATH / Path("pyg_5_layer.txt")
    pyg_4_layer_config_nc = BASE_PATH / Path("pyg_4_layer_no_compute.txt")
    pyg_5_layer_config_nc = BASE_PATH / Path("pyg_5_layer_no_compute.txt")

    pyg_train_trace(pyg_1_layer_config, results_dir / Path("sampling/pyg_1_layer"), overwrite)
    pyg_train_trace(pyg_2_layer_config, results_dir / Path("sampling/pyg_2_layer"), overwrite)
    pyg_train_trace(pyg_3_layer_config, results_dir / Path("sampling/pyg_3_layer"), overwrite)
    pyg_train_trace(pyg_4_layer_config, results_dir / Path("sampling/pyg_4_layer"), overwrite)
    pyg_train_trace(pyg_5_layer_config, results_dir / Path("sampling/pyg_5_layer"), overwrite)
    pyg_train_trace(pyg_4_layer_config_nc, results_dir / Path("sampling/pyg_4_layer_no_compute"), overwrite)
    pyg_train_trace(pyg_5_layer_config_nc, results_dir / Path("sampling/pyg_5_layer_no_compute"), overwrite)



    # dgl_1_layer_config_s = BASE_PATH / Path("dgl_1_layer_only_sample.txt")
    # dgl_2_layer_config_s = BASE_PATH / Path("dgl_2_layer_only_sample.txt")
    # dgl_3_layer_config_s = BASE_PATH / Path("dgl_3_layer_only_sample.txt")
    # dgl_4_layer_config_s = BASE_PATH / Path("dgl_4_layer_only_sample.txt")
    # dgl_5_layer_config_s = BASE_PATH / Path("dgl_5_layer_only_sample.txt")
    #
    # dgl_train_trace(dgl_1_layer_config_s, results_dir / Path("sampling/dgl_1_layer_only_sample"), overwrite)
    # dgl_train_trace(dgl_2_layer_config_s, results_dir / Path("sampling/dgl_2_layer_only_sample"), overwrite)
    # dgl_train_trace(dgl_3_layer_config_s, results_dir / Path("sampling/dgl_3_layer_only_sample"), overwrite)
    # dgl_train_trace(dgl_4_layer_config_s, results_dir / Path("sampling/dgl_4_layer_only_sample"), overwrite)
    # dgl_train_trace(dgl_5_layer_config_s, results_dir / Path("sampling/dgl_5_layer_only_sample"), overwrite)

    # pyg_1_layer_config_s = BASE_PATH / Path("pyg_1_layer_only_sample.txt")
    # pyg_2_layer_config_s = BASE_PATH / Path("pyg_2_layer_only_sample.txt")
    # pyg_3_layer_config_s = BASE_PATH / Path("pyg_3_layer_only_sample.txt")
    # pyg_4_layer_config_s = BASE_PATH / Path("pyg_4_layer_only_sample.txt")
    # pyg_5_layer_config_s = BASE_PATH / Path("pyg_5_layer_only_sample.txt")
    #
    # pyg_train_trace(pyg_1_layer_config_s, results_dir / Path("sampling/pyg_1_layer_only_sample"), overwrite)
    # pyg_train_trace(pyg_2_layer_config_s, results_dir / Path("sampling/pyg_2_layer_only_sample"), overwrite)
    # pyg_train_trace(pyg_3_layer_config_s, results_dir / Path("sampling/pyg_3_layer_only_sample"), overwrite)
    # pyg_train_trace(pyg_4_layer_config_s, results_dir / Path("sampling/pyg_4_layer_only_sample"), overwrite)
    # pyg_train_trace(pyg_5_layer_config_s, results_dir / Path("sampling/pyg_5_layer_only_sample"), overwrite)

