from dataclasses import dataclass
import torch
from marius.tools.preprocess.datasets import *
import marius as m
import marius.tools.configuration.marius_config as mc

import baselines.pyg.datasets as pyg_datasets
import baselines.dgl.datasets as dgl_datasets
from baselines.pyg.helpers import NeighborSampler
import itertools
from omegaconf import OmegaConf
from pathlib import Path
import time
import os
import pandas as pd
import dgl



@dataclass
class SamplingExperimentConfig:
    num_neighbors: list
    incoming: bool
    outgoing: bool
    num_input_nodes: list
    # num_threads: list
    use_hash_map_sets: bool
    run_id: int = 0
    sample_time: float = -1
    map_time: float = -1
    num_edges_sampled: int = -1
    num_nodes_sampled: int = -1



def run_marius_sampler(dataset_dir, configs, output_dir):
    # set dataset configuration
    storage_config = mc.StorageConfig()
    dataset_config = OmegaConf.load(dataset_dir / Path("dataset.yaml"))
    storage_config.dataset = dataset_config
    storage_config.device_type = "cuda"
    storage_config.dataset.base_directory = storage_config.dataset.base_directory + "/"

    edge_storage = m.initializeEdges(storage_config)
    edge_storage.load()
    edge_storage.initializeInMemorySubGraph(torch.tensor([0]))

    results = []
    for config in configs:
        print("Marius running config {}".format(config))
        nbr_sampler = m.LayeredNeighborSampler(edge_storage,
                                               config.num_neighbors,
                                               config.incoming,
                                               config.outgoing,
                                               config.use_hash_map_sets)

        # torch.set_num_threads(config.num_threads)
        input_nodes = torch.randperm(dataset_config.num_nodes)[:config.num_input_nodes]
        if storage_config.device_type == "cuda":
            input_nodes = input_nodes.to(torch.device("cuda"))

        t0 = time.time()
        ret = nbr_sampler.getNeighbors(input_nodes)
        t1 = time.time()

        config.sample_time = t1 - t0
        config.num_nodes_sampled = ret.node_ids.size(0)
        # ret.performMap()
        # t2 = time.time()
        # config.map_time = t2 - t1

        # if config.incoming and config.outgoing:
        #     config.num_edges_sampled = ret.src_sorted_edges.size(0) + ret.dst_sorted_edges.size(0)
        # elif config.incoming and not config.outgoing:
        #     config.num_edges_sampled = ret.dst_sorted_edges.size(0)
        # elif not config.incoming and config.outgoing:
        #     config.num_edges_sampled = ret.src_sorted_edges.size(0)

        results.append(config)

        # write every loop in case of failure
        df = pd.DataFrame(results)
        df.to_csv(output_dir / Path("marius.csv"), float_format="%.4f")



def run_sampling_livejournal(dataset_dir: Path, results_dir: Path, overwrite, enable_dstat, enable_nvidia_smi,
                             show_output, short, num_runs=1):
    dataset_name = "livejournal"
    dataset_dir = dataset_dir / Path(dataset_name)

    if not (dataset_dir / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = livejournal.Livejournal(dataset_dir)
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    output_dir = results_dir / Path("sampling/livejournal")
    # if not results_dir.exists():
    Path(results_dir).mkdir(exist_ok=True)
    Path(results_dir / Path("sampling")).mkdir(exist_ok=True)
    Path(output_dir).mkdir(exist_ok=True)

    num_neighbors_list = [
        # [25, 10],
        # [15, 10, 5],
        # [30, 20, 10],
        [20],
        [20, 20],
        [20, 20, 20],
        [20, 20, 20, 20],
        [20, 20, 20, 20, 20]
    ]
    incoming_list = [False]
    outgoing_list = [True]
    num_input_nodes_list = [10000]
    # num_threads_list = [1, 8, 32]
    run_id_list = [0, 1, 3, 4, 5]
    use_hash_map_sets = [False]

    config_list = itertools.product(num_neighbors_list,
                                    incoming_list,
                                    outgoing_list,
                                    num_input_nodes_list,
                                    # num_threads_list,
                                    run_id_list,
                                    use_hash_map_sets)

    sampling_config_list = []

    for c in config_list:
        sampling_config_list.append(SamplingExperimentConfig(
            num_neighbors=c[0],
            incoming=c[1],
            outgoing=c[2],
            num_input_nodes=c[3],
            # num_threads=c[4],
            run_id=c[4],
            use_hash_map_sets=c[5]
        ))

    run_marius_sampler(dataset_dir, sampling_config_list, output_dir)

    # #    data = pyg_datasets.get_marius_dataset_lp(dataset_dir, add_reverse_edges=False)
    # #    run_pyg_sampler(data, sampling_config_list, output_dir)
    #
    # data = dgl_datasets.get_marius_dataset_nc(dataset_dir.__str__(), add_reverse_edges=False)
    # data = dgl.add_reverse_edges(data, copy_ndata=True, copy_edata=True)
    # data.edata.pop('_ID')
    # data = data.long()
    # data = data.formats('csc')
    #
    # run_dgl_sampler(data, sampling_config_list, output_dir)