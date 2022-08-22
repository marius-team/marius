from marius.tools.preprocess.datasets.ogbn_arxiv import OGBNArxiv
import executor as e
import reporting
from pathlib import Path

BASE_PATH = Path("experiment_manager/example/configs/ogbn_arxiv/")


def run_ogbn_arxiv_disk_cpu(dataset_dir, results_dir, overwrite, enable_dstat, enable_nvidia_smi, show_output, short, num_runs=1):
    """
    Models: GraphSage
    Systems: Marius, DGL, PyG
    """

    dataset_name = "ogbn_arxiv_32"

    marius = BASE_PATH / Path("marius_gs_disk_cpu.yaml")

    if not (dataset_dir / Path(dataset_name) / Path("edges/train_edges.bin")).exists():
        print("==== Preprocessing {} =====".format(dataset_name))
        dataset = OGBNArxiv(dataset_dir / Path(dataset_name))
        dataset.download()
        dataset.preprocess(num_partitions=32, sequential_train_nodes=True)
    else:
        print("==== {} already preprocessed =====".format(dataset_name))

    for i in range(num_runs):
        # Run Marius
        e.run_config(marius, results_dir / Path("ogbn_arxiv/marius_gs_disk_cpu"),
                     overwrite, enable_dstat, enable_nvidia_smi, show_output, i, "marius")


    reporting.print_results_summary([results_dir / Path("ogbn_arxiv/marius_gs_disk_cpu")])
