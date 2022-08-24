from marius.tools.preprocess.datasets.freebase86m import Freebase86m
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/system_comparisons/configs/freebase86m_disk_time")


def run_freebase86m_gs_disk_time(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius
    """

    dataset_name = "freebase86m_disk_time"

    marius_gs = BASE_PATH / Path("marius_gs.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = Freebase86m(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=1024)
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius_gs, results_dir / Path("freebase86m_disk_time/marius_gs"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius", omp_num_threads=4)

    reporting.print_results_summary([results_dir / Path("freebase86m_disk_time/marius_gs")])