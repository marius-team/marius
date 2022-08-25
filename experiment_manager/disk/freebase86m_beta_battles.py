from marius.tools.preprocess.datasets.freebase86m import Freebase86m
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/disk/configs/freebase86m")


def run_freebase86m_beta_battles(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: DistMult, GraphSage, GAT
    Systems: Marius
    """

    dataset_name = "freebase86m_beta_battles"

    dm_mem = BASE_PATH / Path("dm_mem.yaml")
    dm_comet = BASE_PATH / Path("dm_comet.yaml")
    dm_beta = BASE_PATH / Path("dm_beta.yaml")

    gs_mem = BASE_PATH / Path("gs_mem.yaml")
    gs_comet = BASE_PATH / Path("gs_comet.yaml")
    gs_beta = BASE_PATH / Path("gs_beta.yaml")

    gat_mem = BASE_PATH / Path("gat_mem.yaml")
    gat_comet = BASE_PATH / Path("gat_comet.yaml")
    gat_beta = BASE_PATH / Path("gat_beta.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = Freebase86m(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess()
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    # in-memory training
    for i in range(num_runs):
        e.run_config(dm_mem, results_dir / Path("freebase86m_beta_battles/dm_mem"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gs_mem, results_dir / Path("freebase86m_beta_battles/gs_mem"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gat_mem, results_dir / Path("freebase86m_beta_battles/gat_mem"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")

    # comet training
    dataset = Freebase86m(dataset_dir / Path(dataset_name))
    dataset.preprocess(num_partitions=1024)

    for i in range(num_runs):
        e.run_config(dm_comet, results_dir / Path("freebase86m_beta_battles/dm_comet"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gs_comet, results_dir / Path("freebase86m_beta_battles/gs_comet"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gat_comet, results_dir / Path("freebase86m_beta_battles/gat_comet"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")

    # beta training
    dataset = Freebase86m(dataset_dir / Path(dataset_name))
    dataset.preprocess(num_partitions=16)

    for i in range(num_runs):
        e.run_config(dm_beta, results_dir / Path("freebase86m_beta_battles/dm_beta"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gs_beta, results_dir / Path("freebase86m_beta_battles/gs_beta"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")
        e.run_config(gat_beta, results_dir / Path("freebase86m_beta_battles/gat_beta"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")

    reporting.print_results_summary([results_dir / Path("freebase86m_beta_battles/dm_mem"),
                                     results_dir / Path("freebase86m_beta_battles/dm_comet"),
                                     results_dir / Path("freebase86m_beta_battles/dm_beta"),
                                     results_dir / Path("freebase86m_beta_battles/gs_mem"),
                                     results_dir / Path("freebase86m_beta_battles/gs_comet"),
                                     results_dir / Path("freebase86m_beta_battles/gs_beta"),
                                     results_dir / Path("freebase86m_beta_battles/gat_mem"),
                                     results_dir / Path("freebase86m_beta_battles/gat_comet"),
                                     results_dir / Path("freebase86m_beta_battles/gat_beta")])
